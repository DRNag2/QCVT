# -*- coding: utf-8 -*-
"""
Matplotlib rendering of a QICK pulse :class:`~qcvt.model.Schedule`.

The schedule plot shows one horizontal lane per generator/readout channel with
every pulse drawn as a labelled bar on a shared microsecond axis.  An optional
amplitude panel reconstructs the output amplitude vs. time.  Swept parameters
(time, length, gain) are drawn as translucent ranges so you can see, at a glance,
what the loop actually varies before the program is sent to the RFSoC.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .model import Schedule, amplitude_trace, extract_schedule


_GEN_HEIGHT = 0.62
_ADC_HEIGHT = 0.4
_ADC_COLOR = "#1a7a1a"


def _as_schedule(prog_or_schedule) -> Schedule:
    if isinstance(prog_or_schedule, Schedule):
        return prog_or_schedule
    return extract_schedule(prog_or_schedule)


def _channel_colors(gen_chs):
    cmap = plt.cm.tab10
    return {ch: cmap(i % 10) for i, ch in enumerate(gen_chs)}


def _gen_label(sched: Schedule, ch: int, gen_ch_labels, physical_port_labels) -> str:
    label = (gen_ch_labels or {}).get(ch, f"gen {ch}")
    soccfg = sched.soccfg
    if soccfg is not None:
        try:
            dac_id = soccfg["gens"][ch].get("dac")
        except Exception:
            dac_id = None
        if dac_id is not None:
            phys = (physical_port_labels or {}).get(str(dac_id))
            label = f"{label} ({phys or 'dac ' + str(dac_id)})"
    return label


def _adc_label(sched: Schedule, ch: int, physical_port_labels) -> str:
    label = f"ro {ch}"
    soccfg = sched.soccfg
    if soccfg is not None:
        try:
            adc_id = soccfg["readouts"][ch].get("adc")
        except Exception:
            adc_id = None
        if adc_id is not None:
            phys = (physical_port_labels or {}).get(str(adc_id))
            label = f"{label} ({phys or 'adc ' + str(adc_id)})"
    return label


def plot_pulse_schedule(
    prog,
    ax=None,
    max_time_us: Optional[float] = None,
    gen_ch_labels: Optional[dict] = None,
    physical_port_labels: Optional[dict] = None,
    show_readout_triggers: bool = True,
    show_amplitude: bool = False,
    amplitude_units: str = "dac",
    title: Optional[str] = None,
    label_pulses: bool = True,
    schedule: Optional[Schedule] = None,
):
    """Plot a pulse schedule from a compiled QICK ``asm_v2`` program.

    Parameters
    ----------
    prog :
        Compiled program (``AveragerProgramV2``) or a pre-built :class:`Schedule`.
    ax : matplotlib axes, optional
        Axes to draw the schedule on.  If ``None`` a new figure is created (and,
        when ``show_amplitude`` is set, a second amplitude panel is added).
    max_time_us : float, optional
        Right limit of the time axis.  If ``None`` it is inferred from the schedule.
    gen_ch_labels : dict, optional
        Map ``gen_ch (int) -> label`` for lane labels.
    physical_port_labels : dict, optional
        Map RFDC ids (e.g. dac ``'00'``, adc ``'20'``) -> human labels.
    show_readout_triggers : bool
        Draw ADC integration windows as their own lanes.
    show_amplitude : bool
        Add an amplitude-vs-time panel (only when ``ax`` is not supplied).
    amplitude_units : str
        ``"dac"`` (0..maxv) or ``"norm"`` (0..1) for the amplitude panel.
    title : str, optional
        Plot title.
    label_pulses : bool
        Write each pulse's name on its bar.
    schedule : Schedule, optional
        Pre-extracted schedule (avoids re-extraction when plotting repeatedly).

    Returns
    -------
    ax, or (ax, ax_amp)
        ``ax_amp`` is ``None`` when the amplitude panel was not created.
    """
    if amplitude_units not in ("dac", "norm"):
        raise ValueError("amplitude_units must be 'dac' or 'norm'")

    sched = schedule if schedule is not None else _as_schedule(prog)
    ax_amp = None
    want_amp = show_amplitude  # controls the return shape

    if not sched:
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No pulse schedule could be extracted from this program.",
                transform=ax.transAxes, ha="center", va="center")
        if title:
            ax.set_title(title)
        return (ax, None) if want_amp else ax

    owns_figure = ax is None
    draw_amp = show_amplitude
    if show_amplitude and owns_figure:
        _, (ax, ax_amp) = plt.subplots(
            2, 1, figsize=(9, 6), height_ratios=[1.3, 1.0], sharex=True,
            constrained_layout=True,
        )
    elif owns_figure:
        _, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    elif show_amplitude:
        # Caller supplied an axes; we can't safely split it, so skip the panel
        # but still honour the tuple return contract.
        draw_amp = False

    gen_chs = sched.gen_chs
    adc_chs = sched.adc_chs if show_readout_triggers else []
    colors = _channel_colors(gen_chs)

    # Lane layout: generators on top, readouts below.
    y_pos = {}
    idx = 0
    for ch in gen_chs:
        y_pos[("gen", ch)] = idx
        idx += 1
    for ch in adc_chs:
        y_pos[("adc", ch)] = idx
        idx += 1

    draw_lengths = sched.draw_lengths(max_time_us)
    suppressed = sched.suppressed_events()

    # Compute the time window.
    if max_time_us is not None:
        end_us = float(max_time_us)
    else:
        ends = [max(e.t_end, e.t_max + e.len_max) for e in sched.events]
        end_us = max(ends, default=1.0) * 1.03
    end_us = max(end_us, 1e-6)

    # --- generator + readout bars ------------------------------------------------
    for e in sched.gen_events:
        if id(e) in suppressed:
            continue
        y = y_pos.get(("gen", e.ch))
        if y is None:
            continue
        color = colors.get(e.ch, "C0")
        draw_len = draw_lengths.get(id(e), e.length)

        # Sweep ranges drawn behind the nominal bar.
        if e.time_swept:
            ax.barh(y, (e.t_max + draw_len) - e.t_min, left=e.t_min, height=_GEN_HEIGHT,
                    color=color, alpha=0.15, edgecolor="none", zorder=1)
        elif e.length_swept:
            ax.barh(y, e.len_max, left=e.t_start, height=_GEN_HEIGHT,
                    color=color, alpha=0.15, edgecolor="none", zorder=1)

        ax.barh(y, max(draw_len, 0.0), left=e.t_start, height=_GEN_HEIGHT,
                color=color, edgecolor="black", linewidth=0.6, zorder=2,
                hatch="////" if e.periodic else None,
                alpha=0.55 if e.periodic else 1.0)

        if label_pulses:
            label = e.name
            if e.swept_params:
                label += f"\n[sweep: {', '.join(e.swept_params)}]"
            center = e.t_start + max(draw_len, 0.0) / 2.0
            wide_enough = max(draw_len, 0.0) > 0.08 * end_us
            if wide_enough:
                ax.text(center, y, label, ha="center", va="center",
                        fontsize=7, color="white", zorder=3, fontweight="bold")
            else:
                ax.text(e.t_start, y + _GEN_HEIGHT / 2 + 0.02, label, ha="left",
                        va="bottom", fontsize=6.5, color=color, zorder=3)

    for e in sched.adc_events:
        y = y_pos.get(("adc", e.ch))
        if y is None:
            continue
        ax.barh(y, max(e.length, 0.01), left=e.t_start, height=_ADC_HEIGHT,
                color=_ADC_COLOR, alpha=0.7, edgecolor="black", linewidth=0.8, zorder=2)

    # --- axis cosmetics ----------------------------------------------------------
    y_ticks, y_labels = [], []
    for ch in gen_chs:
        y_ticks.append(y_pos[("gen", ch)])
        lab = _gen_label(sched, ch, gen_ch_labels, physical_port_labels)
        freqs = {round(e.freq, 3) for e in sched.gen_events if e.ch == ch and e.freq is not None}
        if len(freqs) == 1:
            lab += f"\n{next(iter(freqs)):g} MHz"
        y_labels.append(lab)
    for ch in adc_chs:
        y_ticks.append(y_pos[("adc", ch)])
        y_labels.append(_adc_label(sched, ch, physical_port_labels))

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_ylim(-0.6, len(y_pos) - 0.4)
    ax.set_xlim(0, end_us)
    ax.set_xlabel("Time (µs)")
    ax.grid(True, axis="x", alpha=0.3)
    if title:
        ax.set_title(title, fontsize=11)

    legend_items = []
    if any(e.periodic for e in sched.gen_events):
        legend_items.append(Patch(facecolor="0.6", hatch="////", alpha=0.55,
                                   edgecolor="black", label="periodic (CW)"))
    if adc_chs:
        legend_items.append(Patch(facecolor=_ADC_COLOR, alpha=0.7,
                                   edgecolor="black", label="ADC integration"))
    if any(e.swept_params for e in sched.gen_events):
        legend_items.append(Patch(facecolor="0.6", alpha=0.15, edgecolor="none",
                                   label="swept range"))
    if legend_items:
        ax.legend(handles=legend_items, loc="upper right", fontsize=7, framealpha=0.9)

    if draw_amp and ax_amp is not None:
        _draw_amplitude_panel(ax_amp, sched, colors, draw_lengths,
                              amplitude_units, end_us, gen_ch_labels)

    return (ax, ax_amp) if want_amp else ax


def _draw_amplitude_panel(ax_amp, sched: Schedule, colors, draw_lengths,
                          amplitude_units, end_us, gen_ch_labels):
    dac_units = amplitude_units == "dac"
    prog = sched.prog
    seen = set()
    for e in sched.gen_events:
        draw_len = draw_lengths.get(id(e), e.length)
        color = colors.get(e.ch, "C0")
        label = (gen_ch_labels or {}).get(e.ch, f"gen {e.ch}")
        legend_label = label if e.ch not in seen else "_nolegend_"

        if e.gain_swept:
            t_lo, a_lo = amplitude_trace(prog, e, length_us=draw_len,
                                         dac_units=dac_units, gain_override=e.gain_min)
            t_hi, a_hi = amplitude_trace(prog, e, length_us=draw_len,
                                         dac_units=dac_units, gain_override=e.gain_max)
            if t_lo is not None and t_hi is not None and np.array_equal(t_lo, t_hi):
                ax_amp.fill_between(t_lo, np.abs(a_lo), np.abs(a_hi), color=color,
                                    alpha=0.3, linewidth=0, label="_nolegend_")
                ax_amp.plot(t_lo, (np.abs(a_lo) + np.abs(a_hi)) / 2, color=color,
                            linewidth=2.0, label=legend_label)
                seen.add(e.ch)
                continue

        t_arr, amp = amplitude_trace(prog, e, length_us=draw_len, dac_units=dac_units)
        if t_arr is None:
            continue
        ax_amp.plot(t_arr, np.abs(amp), color=color, linewidth=2.0, label=legend_label)
        seen.add(e.ch)

    for e in sched.adc_events:
        ax_amp.axvspan(e.t_start, e.t_end, color=_ADC_COLOR, alpha=0.2, lw=0)

    ax_amp.set_xlim(0, end_us)
    ax_amp.set_ylim(bottom=0)
    ax_amp.set_xlabel("Time (µs)")
    ax_amp.set_ylabel("Amplitude (DAC units)" if dac_units else "Amplitude (norm)")
    ax_amp.grid(True, alpha=0.3)
    handles, labels = ax_amp.get_legend_handles_labels()
    if sched.adc_events:
        handles.append(Patch(facecolor=_ADC_COLOR, alpha=0.2, edgecolor="none"))
        labels.append("ADC integration")
    if handles:
        ax_amp.legend(handles=handles, labels=labels, loc="upper right",
                      fontsize=7, framealpha=0.9)


def show_schedule(
    prog,
    title: str = "Pulse schedule",
    show_amplitude: bool = True,
    amplitude_units: str = "dac",
    gen_ch_labels: Optional[dict] = None,
    physical_port_labels: Optional[dict] = None,
) -> None:
    """Quickly display a pulse schedule interactively (no files saved).

    Intended for a fast look while running experiments, e.g. right before sending
    a program to the RFSoC.
    """
    plot_pulse_schedule(
        prog,
        show_amplitude=show_amplitude,
        amplitude_units=amplitude_units,
        gen_ch_labels=gen_ch_labels,
        physical_port_labels=physical_port_labels,
        title=title,
    )
    plt.show()
