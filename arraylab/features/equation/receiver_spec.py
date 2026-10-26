"""Shared acquisition and output-grid specifications, without signal simulation.

This module contains timing/axis contracts only. It does not calculate a link
budget, synthesize a target, or construct the receiver's linear coefficients.
"""
from __future__ import annotations

import math
import numpy as np

C = 299792458.0


def receiver_filter_spec(config):
    w = config["waveform"]
    ratio = float(w["rx_bandwidth_hz"])/float(w["fs_hz"])*(2 if w["kind"] == "fmcw" else 1)
    count = 1 if ratio >= 1-1e-12 else 33
    return {"normalized_cutoff": ratio, "filter_taps": count,
            "filter_history_samples": (count-1)//2}


def receiver_layout(config):
    w, a = config["waveform"], config["array"]
    fs, pri = float(w["fs_hz"]), float(w["pri_s"])
    n, nr, nv = int(w["samples"]), int(w["range_fft"]), int(w["doppler_fft"])
    spec = receiver_filter_spec(config)
    half = spec["filter_history_samples"]
    start = float(w["adc_start_s"]) if w["kind"] == "fmcw" else 0.
    stride = int(a["tx_count"]) if a["mode"] in ("tdm", "cdm") else 1
    if w["kind"] == "fmcw":
        slope = float(w["bandwidth_hz"])/float(w["ramp_s"])
        step = C*fs/(2*slope*nr)
        final = min((nr-1)//2, math.floor(float(w["range_max_m"])*(1+1e-12)/step))
        range_max = final*step
    else:
        range_max = min(float(w["range_max_m"]), C*(n-1)/(2*fs))
    velocity_step = C/(2*float(w["fc_hz"])*nv*pri*stride)
    return {**spec, "adc_start_s": start, "adc_last_s": start+(n-1)/fs,
            "input_first_s": start-half/fs, "input_last_s": start+(n-1+half)/fs,
            "input_samples": n+2*half, "input_record_duration_s": (n+2*half)/fs,
            "receive_end_s": min(n/fs, pri),
            "processed_range_max_m": range_max,
            "processed_velocity_min_mps": -(nv//2)*velocity_step,
            "processed_velocity_max_mps": ((nv-1)//2)*velocity_step,
            "doppler_stride": stride}


def far_field_spec(config, tx_distance_m, rx_distance_m):
    """Fraunhofer guard for supplied channel coordinates about radar origin.

    Element apertures are not provided, so this is a necessary guard on known
    geometry, not a complete antenna near-field certification.
    """
    extents = {}
    for side in ("tx", "rx"):
        points = np.vstack((np.zeros((1, 3)), np.asarray(config["array"][side+"_positions_m"], float)))
        extents[side] = float(np.max(np.linalg.norm(points[:, None, :]-points[None, :, :], axis=2)))
    w = config["waveform"]
    highest_rf = float(w["fc_hz"])+(float(w["bandwidth_hz"])/2 if w["kind"] in ("fmcw", "lfm") else 0.)
    wavelength = C/highest_rf
    required = {side:2*extent**2/wavelength for side,extent in extents.items()}
    distances = {"tx":float(tx_distance_m), "rx":float(rx_distance_m)}
    covered = all(distances[side] >= required[side]*(1-1e-12) for side in ("tx", "rx"))
    return {"known_geometry_only":True, "valid":bool(covered),
        "coordinate_reference":"channel coordinates plus radar reference origin; no antenna-element aperture supplied",
        "shortest_wavelength_m":wavelength,
        **{side+"_coordinate_extent_m":extents[side] for side in ("tx", "rx")},
        **{side+"_required_distance_m":required[side] for side in ("tx", "rx")},
        **{side+"_path_distance_m":distances[side] for side in ("tx", "rx")}}


def observable_path(config, delay_s, doppler_hz):
    """Whether one path fits the actual acquisition and selectable output axes."""
    w = config["waveform"]
    layout = receiver_layout(config)
    delay, fd = float(delay_s), float(doppler_hz)
    limit_fd = 1/(2*float(w["pri_s"])*layout["doppler_stride"])
    unambiguous = -limit_fd <= fd < limit_fd
    velocity = fd*C/(2*float(w["fc_hz"]))
    # Allow only round-off at generated axis endpoints, not a half-bin extension.
    vtol = max(1e-12, abs(layout["processed_velocity_min_mps"])*1e-12)
    velocity_grid = layout["processed_velocity_min_mps"]-vtol <= velocity <= layout["processed_velocity_max_mps"]+vtol
    result = {"delay_s": delay, "unambiguous_doppler": unambiguous,
              "within_velocity_grid": velocity_grid,
              "processed_range_max_m": layout["processed_range_max_m"],
              "processed_velocity_min_mps": layout["processed_velocity_min_mps"],
              "processed_velocity_max_mps": layout["processed_velocity_max_mps"]}
    if w["kind"] == "fmcw":
        slope = float(w["bandwidth_hz"])/float(w["ramp_s"])
        beat = slope*delay+fd
        if_limit = min(float(w["rx_bandwidth_hz"]), float(w["fs_hz"])/2)
        full = layout["adc_start_s"] >= delay and layout["adc_last_s"]-delay < float(w["ramp_s"])
        settled = layout["input_first_s"] >= delay and layout["input_last_s"]-delay < float(w["ramp_s"])
        apparent = C*beat/(2*slope)
        in_range = 0 <= apparent <= layout["processed_range_max_m"]*(1+1e-12)
        result.update(beat_hz=beat, if_limit_hz=if_limit, apparent_range_m=apparent,
                      within_if=0 <= beat < if_limit)
    else:
        opening = float(w["pulse_width_s"])+float(w["recovery_s"])
        width = float(w["pulse_width_s"])
        tail = layout["filter_history_samples"]/float(w["fs_hz"])
        full = delay >= opening and delay+width <= layout["receive_end_s"]
        settled = delay-tail >= opening and delay+width+tail <= layout["receive_end_s"]
        in_range = 0 <= delay*C/2 <= layout["processed_range_max_m"]*(1+1e-12)
        result.update(receive_open_s=opening, receive_end_s=layout["receive_end_s"], within_if=True)
    physical = bool(full and settled and unambiguous and result["within_if"])
    grid = bool(velocity_grid and in_range)
    result.update(complete_echo=full, filter_settled=settled, within_processed_range=in_range,
                  physical_observable=physical, within_processed_grid=grid,
                  observable=physical and grid)
    return result
