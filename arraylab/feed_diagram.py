"""Self-contained electrical connection diagrams, deliberately not physical layouts."""
from html import escape


def section_lengths(section):
    """Add display-only defaults to legacy snapshots without mutating them."""
    item = dict(section)
    item.setdefault("matched_delay_deg", 0.0)
    item.setdefault("matched_delay_ohm", item["child_equivalent_ohm"])
    item.setdefault("total_electrical_length_deg", item["electrical_length_deg"] + item["matched_delay_deg"])
    return item


def port_lengths(report):
    """Describe stored source-to-port paths; no solver or current inputs are used."""
    incoming = {line["child"]: section_lengths(line) for line in report["sections"]}
    result = []
    for port in report["ports"]:
        node, seen = f"P{port['port']}", set()
        transform, delay = 0.0, 0.0
        while node != report["root"]:
            if node in seen or node not in incoming:
                raise ValueError("馈电快照的端口路径不完整。")
            seen.add(node)
            line = incoming[node]
            transform += line["electrical_length_deg"]
            delay += line["matched_delay_deg"]
            node = line["parent"]
        item = dict(port)
        item.setdefault("transform_path_deg", transform)
        item.setdefault("branch_delay_path_deg", delay)
        item.setdefault("network_electrical_length_deg", transform + delay)
        item.setdefault("total_electrical_length_deg", transform + delay + item.get("applied_delay_deg", 0.0))
        result.append(item)
    return result


def diagram_svg(report):
    serial = report["settings"]["topology"] == "series"
    symmetric = report["settings"]["topology"] == "symmetric_series"
    n = report["metrics"]["ports"]
    max_depth = max(node["depth"] for node in report["nodes"])
    positions = {}
    for node in report["nodes"]:
        if symmetric:
            if node["id"] == report["root"]:
                x, y = 1400, 125
            else:
                left = node["arm"] == "left"
                if node["kind"] == "port":
                    port = node["first_port"]
                    x = 80 if left else 2720
                    step = n // 2 - port if left else port - n // 2 - 1
                else:
                    x = 740 if left else 2060
                    step = node["depth"] - 1
                y = 385 + step * 220
        elif serial and n > 1:
            x = 760 if node["kind"] == "port" else 100
            y = 125 + (node["first_port"] - 1) * 220
        else:
            x = 100 + (max_depth if node["kind"] == "port" else node["depth"]) * 660
            y = 125 + ((node["first_port"] + node["last_port"]) / 2 - 1) * 140
        positions[node["id"]] = (x, y)
    width = max(x for x, y in positions.values()) + 120
    height = max(y for x, y in positions.values()) + 90
    title = "1:2 split + symmetric series feed" if symmetric else "Series tapped feed" if serial else "Corporate parallel feed"
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#f7fafc"/>',
             '<g font-family="Arial, sans-serif" font-size="12" fill="#25465c">',
             f'<text x="20" y="26" font-size="18">{title} / {n} ports / {report["array_parameters"]["frequency_ghz"]:g} GHz</text>',
             '<text x="20" y="49">T: transformer | D: matched branch delay | C: terminal compensation. Electrical degrees at design frequency.</text>',
             '<text x="20" y="69">Each T and D uses its own impedance. Zero-degree D/C means bypass. Not to scale.</text>']
    port_map = {f"P{port['port']}": port for port in report["ports"]}

    def segment(points, color="#297da7"):
        parts.append(f'<polyline points="{points}" stroke="{color}" stroke-width="2" fill="none"/>')

    def block(x, y, label, impedance, degrees, kind, width=136, outside=False):
        fill, stroke = {"T": ("#e4edf8", "#297da7"), "D": ("#e6f5ed", "#27815a"),
                        "C": ("#fff0d9", "#ac731d")}[kind]
        if outside:
            fill, stroke = "#fff0d9", "#b66b20"
        parts.append(f'<rect x="{x-width/2}" y="{y-23}" width="{width}" height="46" rx="3" fill="{fill}" stroke="{stroke}"/>')
        parts.append(f'<text x="{x}" y="{y-5}" text-anchor="middle">{escape(label)}: {impedance:.5g} ohm</text>')
        parts.append(f'<text x="{x}" y="{y+13}" text-anchor="middle">{degrees:.7g} deg</text>')

    for original in report["sections"]:
        line = section_lengths(original)
        x1, y1 = positions[line["parent"]]
        x2, y2 = positions[line["child"]]
        color = "#b66b20" if line["outside_range"] else "#297da7"
        port = port_map.get(line["child"])
        has_compensation = port is not None and "applied_delay_deg" in port
        label_t, label_d = line["id"], "D" + line["id"][1:]
        if x1 == x2:
            # Series backbone: enough separation to draw both actual segments.
            segment(f"{x1},{y1+22} {x2},{y2-22}", color)
            block(x1, y1 + 68, label_t, line["line_ohm"], line["electrical_length_deg"], "T", outside=line["outside_range"])
            block(x1, y2 - 68, label_d, line["matched_delay_ohm"], line["matched_delay_deg"], "D")
        else:
            direction = 1 if x2 > x1 else -1
            # Runs hold separate T/D/C blocks in source-to-load order.
            if symmetric and line["parent"] == report["root"]:
                run_y = y1 + 130
                start = x1 + 90 * direction
                end = x2
                segment(f"{x1+53*direction},{y1} {start},{y1} {start},{run_y} {end},{run_y} {x2},{y2-22}", color)
            else:
                start = x1 + 90 * direction
                end = x2 - 53 * direction
                run_y = y2
                segment(f"{x1+53*direction},{y1} {start},{y1} {start},{run_y} {end},{run_y}", color)
            count = 3 if has_compensation else 2
            centers = [start + (end-start) * (index+.5)/count for index in range(count)]
            block(centers[0], run_y, label_t, line["line_ohm"], line["electrical_length_deg"], "T", outside=line["outside_range"])
            block(centers[1], run_y, label_d, line["matched_delay_ohm"], line["matched_delay_deg"], "D")
            if has_compensation:
                block(centers[2], run_y, f"C{port['port']}", port["compensation_line_ohm"], port["applied_delay_deg"], "C")
    for node in report["nodes"]:
        x, y = positions[node["id"]]
        fill = "#e8f3f6" if node["kind"] == "port" else "#e4edf8"
        parts.append(f'<rect x="{x-53}" y="{y-22}" width="106" height="44" rx="4" fill="{fill}" stroke="#8ca6ba"/>')
        parts.append(f'<text x="{x}" y="{y-4}" text-anchor="middle">{escape(node["id"])}</text>')
        parts.append(f'<text x="{x}" y="{y+13}" text-anchor="middle">R = {node["equivalent_ohm"]:.4g} ohm</text>')
    parts.append('</g></svg>')
    return "\n".join(parts)
