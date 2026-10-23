"""Self-contained SVG connection diagrams; deliberately not physical layouts."""
from html import escape


def diagram_svg(report):
    serial = report["settings"]["topology"] == "series"
    symmetric = report["settings"]["topology"] == "symmetric_series"
    n = report["metrics"]["ports"]
    max_depth = max(node["depth"] for node in report["nodes"])
    positions = {}
    for node in report["nodes"]:
        if symmetric:
            if node["id"] == report["root"]:
                x, y = 730, 100
            else:
                left = node["arm"] == "left"
                if node["kind"] == "port":
                    port = node["first_port"]
                    x = 90 if left else 1370
                    step = n // 2 - port if left else port - n // 2 - 1
                else:
                    x = 430 if left else 1030
                    step = node["depth"] - 1
                y = 260 + step * 100
        elif serial and n > 1:
            if node["kind"] == "port":
                x, y = 570, 100 + (node["first_port"] - 1) * 100
            else:
                x, y = 90, 100 + (node["first_port"] - 1) * 100
        else:
            x = 90 + (max_depth if node["kind"] == "port" else node["depth"]) * 340
            y = 100 + ((node["first_port"] + node["last_port"]) / 2 - 1) * 90
        positions[node["id"]] = (x, y)
    width = max(x for x, y in positions.values()) + 120
    height = max(y for x, y in positions.values()) + 100
    title = "1:2 split + symmetric series feed" if symmetric else "Series tapped feed" if serial else "Corporate parallel feed"
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#f7fafc"/>',
             '<g font-family="Arial, sans-serif" font-size="13" fill="#25465c">',
             f'<text x="20" y="26" font-size="18">{title} / {n} ports / {report["array_parameters"]["frequency_ghz"]:g} GHz</text>',
             '<text x="20" y="49">Divider: 90 deg. Terminal phase correction: port table. Not to scale.</text>']
    for line in report["sections"]:
        x1, y1 = positions[line["parent"]]
        x2, y2 = positions[line["child"]]
        color = "#b66b20" if line["outside_range"] else "#297da7"
        anchor = "start"
        if symmetric and line["parent"] == report["root"]:
            direction = 1 if x2 > x1 else -1
            points = f"{x1 + 53 * direction},{y1} {x2},{y1} {x2},{y2 - 22}"
            tx, ty = (x1 + x2) / 2, y1 - 12
            anchor = "middle"
        elif x1 == x2:
            points = f"{x1},{y1 + 22} {x2},{y2 - 22}"
            tx, ty = x1 + 18, (y1 + y2) / 2 + 5
        else:
            direction = 1 if x2 > x1 else -1
            bend = x1 + (x2 - x1) * .45
            points = f"{x1 + 53 * direction},{y1} {bend},{y1} {bend},{y2} {x2 - 53 * direction},{y2}"
            tx, ty = bend + 8 * direction, y2 - 12
            anchor = "start" if direction > 0 else "end"
        parts.append(f'<polyline points="{points}" stroke="{color}" stroke-width="2" fill="none"/>')
        parts.append(f'<text x="{tx}" y="{ty}" text-anchor="{anchor}" fill="{color}">{line["id"]}: {line["line_ohm"]:.3f} ohm</text>')
    for node in report["nodes"]:
        x, y = positions[node["id"]]
        fill = "#e8f3f6" if node["kind"] == "port" else "#e4edf8"
        parts.append(f'<rect x="{x-53}" y="{y-22}" width="106" height="44" rx="4" fill="{fill}" stroke="#8ca6ba"/>')
        parts.append(f'<text x="{x}" y="{y-4}" text-anchor="middle">{escape(node["id"])}</text>')
        parts.append(f'<text x="{x}" y="{y+13}" text-anchor="middle">R = {node["equivalent_ohm"]:.4g} ohm</text>')
    parts.append('</g></svg>')
    return "\n".join(parts)
