# -*- coding: utf-8 -*-
"""
CSV / NPZ / table exports derived from a QICK pulse :class:`~qcvt.model.Schedule`.

Two kinds of export are provided:

* :func:`export_amplitude_traces_csv` -- amplitude vs. time sampled exactly at
  pulse edges (plus an NPZ with the raw arrays).
* :func:`export_edge_matrices_csv` -- compact "edge matrices": one row per lane,
  one column per timestamp at which some lane's on/off state changes.

:func:`csv_to_table_png` renders any of these CSVs as a highlighted table image.
"""

from __future__ import annotations

import csv as _csv
from typing import List, Optional, Tuple

import numpy as np

from .model import Schedule, amplitude_trace, extract_schedule


def _as_schedule(prog_or_schedule) -> Schedule:
    if isinstance(prog_or_schedule, Schedule):
        return prog_or_schedule
    return extract_schedule(prog_or_schedule)


def _gen_intervals(sched: Schedule, window_end_us: float, dac_units: bool):
    """Return ``{ch: [(t0, t1, amp), ...]}`` for generator channels."""
    draw_lengths = sched.draw_lengths(window_end_us)
    suppressed = sched.suppressed_events()
    out = {}
    for e in sched.gen_events:
        if id(e) in suppressed:
            continue
        draw_len = draw_lengths.get(id(e), e.length)
        t_arr, amp_arr = amplitude_trace(sched.prog, e, length_us=draw_len, dac_units=dac_units)
        if t_arr is None or amp_arr is None:
            continue
        amp = float(np.nanmax(np.abs(amp_arr))) if amp_arr.size else 0.0
        if amp == 0.0:
            continue
        out.setdefault(e.ch, []).append((float(e.t_start), float(e.t_start + draw_len), amp))
    return out


def _adc_intervals(sched: Schedule):
    out = {}
    for e in sched.adc_events:
        out.setdefault(e.ch, []).append((float(e.t_start), float(e.t_end), 1.0))
    return out


def export_amplitude_traces_csv(
    prog,
    csv_path: str,
    t0_us: float,
    t1_us: Optional[float],
    amplitude_units: str = "dac",
    schedule: Optional[Schedule] = None,
) -> str:
    """Export per-channel amplitude vs. time to CSV (and a companion ``.npz``).

    Amplitudes are sampled on the union of all pulse edge times in
    ``[t0_us, t1_us]`` so piecewise-constant pulses are represented exactly.
    Returns the CSV path.
    """
    if amplitude_units not in ("dac", "norm"):
        raise ValueError("amplitude_units must be 'dac' or 'norm'")

    sched = schedule if schedule is not None else _as_schedule(prog)
    if not sched:
        raise RuntimeError("No schedule could be extracted from this program.")
    if t1_us is None:
        t1_us = sched.end_us()

    dac_units = amplitude_units == "dac"
    intervals = _gen_intervals(sched, t1_us, dac_units)
    gen_chs = sorted(intervals)
    if not gen_chs:
        raise RuntimeError("No generator pulses found in schedule.")

    edge_times = {float(t0_us), float(t1_us)}
    for segs in intervals.values():
        for a, b, _amp in segs:
            if t0_us <= a <= t1_us:
                edge_times.add(float(a))
            if t0_us <= b <= t1_us:
                edge_times.add(float(b))
    times = np.array(sorted(edge_times), dtype=float)

    amp_mat = np.zeros((times.size, len(gen_chs)), dtype=float)
    for j, ch in enumerate(gen_chs):
        col = np.zeros(times.size, dtype=float)
        for t0, t1, amp in intervals[ch]:
            mask = (times >= t0) & (times < t1)
            col[mask] = np.maximum(col[mask], amp)
        amp_mat[:, j] = col

    with open(csv_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["time_us"] + [f"gen_{ch}" for ch in gen_chs])
        for i, t in enumerate(times):
            w.writerow([f"{t:.9f}"] + [f"{amp_mat[i, j]:.9f}" for j in range(len(gen_chs))])

    npz_path = csv_path.rsplit(".", 1)[0] + ".npz"
    np.savez(npz_path, time_us=times, gen_chs=np.array(gen_chs, dtype=int), amp=amp_mat)
    return csv_path


def export_edge_matrices_csv(
    prog,
    out_prefix: str,
    t0_us: float,
    t1_us: Optional[float],
    rows: Optional[List[Tuple[str, str, int]]] = None,
    amplitude_units: str = "dac",
    schedule: Optional[Schedule] = None,
) -> Tuple[str, str]:
    """Export state and amplitude "edge matrices" as two CSVs.

    * ``{out_prefix}_state.csv`` -- entries are ``on`` / ``off``.
    * ``{out_prefix}_amp.csv``   -- entries are the amplitude (0 when off).

    Columns are timestamps (ns) at which at least one lane changes state.

    ``rows`` is a list of ``(label, kind, ch)`` with ``kind`` in ``{"gen","adc"}``;
    when ``None`` it defaults to every generator then every readout channel.
    """
    if amplitude_units not in ("dac", "norm"):
        raise ValueError("amplitude_units must be 'dac' or 'norm'")

    sched = schedule if schedule is not None else _as_schedule(prog)
    if not sched:
        raise RuntimeError("No schedule could be extracted from this program.")
    if t1_us is None:
        t1_us = sched.end_us()

    dac_units = amplitude_units == "dac"
    gen_intervals = _gen_intervals(sched, t1_us, dac_units)
    adc_intervals = _adc_intervals(sched)
    intervals = {("gen", ch): segs for ch, segs in gen_intervals.items()}
    intervals.update({("adc", ch): segs for ch, segs in adc_intervals.items()})

    if rows is None:
        rows = ([(f"gen {ch}", "gen", ch) for ch in sorted(gen_intervals)]
                + [(f"ro {ch}", "adc", ch) for ch in sorted(adc_intervals)])

    def state_amp_at(kind: str, ch: int, t: float) -> Tuple[bool, float]:
        on, amp = False, 0.0
        for t0, t1, a in intervals.get((kind, ch), []):
            if t0 <= t < t1:
                on = True
                amp = max(amp, float(a))
        return on, amp

    edge_times = {float(t0_us), float(t1_us)}
    for segs in intervals.values():
        for a, b, _amp in segs:
            if t0_us <= a <= t1_us:
                edge_times.add(float(a))
            if t0_us <= b <= t1_us:
                edge_times.add(float(b))
    edge_times = sorted(edge_times)

    # Keep only timestamps where some lane's on/off state actually changes.
    columns: List[float] = []
    for t in edge_times:
        if not columns:
            columns.append(t)
            continue
        prev = columns[-1]
        changed = any(
            state_amp_at(kind, ch, prev)[0] != state_amp_at(kind, ch, t)[0]
            for _label, kind, ch in rows
        )
        if changed:
            columns.append(t)

    state_rows, amp_rows = [], []
    for label, kind, ch in rows:
        unit = "" if kind == "adc" else (" (DAC units)" if dac_units else " (norm)")
        srow = [label]
        arow = [label + (" (ADC gate)" if kind == "adc" else unit)]
        for col in columns:
            on, amp = state_amp_at(kind, int(ch), float(col))
            srow.append("on" if on else "off")
            arow.append(f"{amp:.6g}" if on else "0")
        state_rows.append(srow)
        amp_rows.append(arow)

    header = ["timestamp (ns)"] + _unique_time_labels([c * 1_000.0 for c in columns])
    state_path = f"{out_prefix}_state.csv"
    amp_path = f"{out_prefix}_amp.csv"
    for path, data in ((state_path, state_rows), (amp_path, amp_rows)):
        with open(path, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(header)
            w.writerows(data)
    return state_path, amp_path


def _unique_time_labels(values_ns) -> List[str]:
    """Format timestamps with the fewest decimals that keep them distinct."""
    for decimals in range(2, 10):
        labels = []
        for v in values_ns:
            mant, exp = f"{v:.{decimals}e}".split("e")
            mant = mant.rstrip("0").rstrip(".")
            labels.append(f"{mant}e{int(exp)}")
        if len(labels) == len(set(labels)):
            return labels
    seen, out = {}, []
    for lbl in labels:
        n = seen.get(lbl, 0) + 1
        seen[lbl] = n
        out.append(lbl if n == 1 else f"{lbl}({n})")
    return out


def csv_to_table_png(csv_path: str, png_path: str, title: str = "") -> None:
    """Render a CSV (e.g. an edge matrix) as a PNG table.

    Cells that are ``on`` or have a numeric value > 0 are highlighted.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)

    def _display_col(col: str) -> str:
        if col == df.columns[0]:
            return col
        s = str(col).strip()
        if "(" in s and s.endswith(")"):
            base, suf = s.rsplit("(", 1)
            base = base.strip()
            try:
                float(base)
                return base + "(" + suf
            except Exception:
                pass
        return s

    display_cols = [_display_col(c) for c in df.columns]
    fig_h = max(2.5, 0.55 * (len(df) + 1))
    fig_w = max(8.0, 0.8 * len(df.columns))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=display_cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.35)
    try:
        first_w = tbl[(0, 0)].get_width()
        for (r, c), cell in tbl.get_celld().items():
            if c == 0:
                cell.set_width(first_w * 1.8)
    except Exception:
        pass

    highlight = "#d9ecff"
    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c <= 0:
            continue
        try:
            val = df.iat[r - 1, c]
        except Exception:
            continue
        on = False
        if isinstance(val, str):
            v = val.strip().lower()
            on = v == "on"
            if not on:
                try:
                    on = float(v) > 0
                except Exception:
                    on = False
        else:
            try:
                on = float(val) > 0
            except Exception:
                on = False
        if on:
            cell.set_facecolor(highlight)

    if title:
        ax.set_title(title, pad=6)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
