# -*- coding: utf-8 -*-
"""
Pulse visualizer for QICK asm_v2 programs (QCVT).

Use without RFSoC connection in two ways:

1. From a saved compiled program pickle (e.g. from a past measurement):
   - Load the pickle and pass the program to plot_pulse_schedule(prog, ...).

2. From program class + config + soccfg from file (offline build):
   - Save soccfg once when connected: save_soccfg_to_json(soc, path).
   - Load soccfg from file: soccfg = load_soccfg_from_json(path).
   - Build program: prog = YourProgram(soccfg, reps=1, cfg=config).
   - Plot: plot_pulse_schedule(prog, ...).
"""

from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Optional, List, Tuple, Any

# Use serif font for all pulse schedule plots
plt.rcParams["font.family"] = "serif"

# Optional: avoid importing qick if only using saved pickle + plotting
try:
    from qick.qick_asm import QickConfig
except ImportError:
    QickConfig = None


def save_soccfg_to_json(soc, path: str) -> None:
    """
    Save the current RFSoC config to a JSON file so you can build and visualize
    programs without being connected.

    Call this once while connected, e.g.:
        save_soccfg_to_json(soc, 'qick_config.json')

    Then offline:
        from qcvt import load_soccfg_from_json, plot_pulse_schedule
        soccfg = load_soccfg_from_json('qick_config.json')
        prog = Time_of_flight(soccfg, reps=2, cfg=config)
        plot_pulse_schedule(prog)
    """
    cfg = soc.get_cfg()
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("Saved soccfg to %s" % path)


def load_soccfg_from_json(path: str):
    """
    Load a QickConfig from a JSON file (saved earlier with save_soccfg_to_json).
    Use this to build programs offline for visualization or testing.
    """
    if QickConfig is None:
        raise ImportError("qick is required for load_soccfg_from_json")
    return QickConfig(path)


def _scalar_value(x) -> float:
    """Get a scalar float from a QickParam or number."""
    if hasattr(x, "start"):
        return float(x.start)
    if hasattr(x, "minval"):
        return float(x.minval())
    return float(x)


def _extract_schedule(prog) -> List[Tuple[int, str, int, int, str]]:
    """
    Extract (ch, name, t_cycles, length_cycles, kind) from a compiled program.

    kind:
      - 'gen': generator pulse on DAC output
      - 'adc': ADC integration window (from Trigger macro ros + width)
    """
    schedule = []
    try:
        macro_list = getattr(prog, "macro_list", [])
        pulses = getattr(prog, "pulses", {})
        soccfg = getattr(prog, "soccfg", None)
        if not macro_list or not pulses or soccfg is None:
            return schedule

        for macro in macro_list:
            cname = type(macro).__name__
            if cname == "Pulse":
                ch = getattr(macro, "ch", None)
                name = getattr(macro, "name", None)
                if ch is None or name is None or name not in pulses:
                    continue
                t_regs = getattr(macro, "t_regs", {})
                t_cycles_raw = t_regs.get("t")
                if t_cycles_raw is None:
                    continue
                t_cycles = _scalar_value(t_cycles_raw)
                if np.isnan(t_cycles) or np.isinf(t_cycles):
                    continue
                pulse = pulses[name]
                length_param = pulse.get_length()
                length_cycles = _scalar_value(length_param)
                if np.isnan(length_cycles) or np.isinf(length_cycles) or length_cycles < 0:
                    continue
                schedule.append((int(ch), str(name), int(round(t_cycles)), int(round(length_cycles)), "gen"))
            elif cname == "Trigger":
                t_regs = getattr(macro, "t_regs", {})
                t_cycles_raw = t_regs.get("t")
                ros = getattr(macro, "ros", None) or []
                if t_cycles_raw is not None and ros:
                    t_cycles = _scalar_value(t_cycles_raw)
                    width_raw = t_regs.get("width")
                    if width_raw is not None:
                        width_cycles = _scalar_value(width_raw)
                        if not (np.isnan(t_cycles) or np.isnan(width_cycles) or np.isinf(t_cycles) or np.isinf(width_cycles) or width_cycles < 0):
                            for ro in ros:
                                schedule.append((int(ro), "adc", int(round(t_cycles)), int(round(width_cycles)), "adc"))
    except Exception as e:
        import warnings
        warnings.warn("Could not extract full schedule from program: %s" % e)
    return schedule


def _get_pulse_amplitude_trace(prog, ch: int, name: str, t_us: float, length_us: float, dac_units: bool = True):
    """
    Return (t_us_array, amplitude_array) for one pulse for amplitude plotting.

    If dac_units is True, amplitude is in DAC units (0 to maxv, typically 32766).
    If False, amplitude is normalized 0–1 (gain * envelope).
    """
    pulses = getattr(prog, "pulses", {})
    soccfg = getattr(prog, "soccfg", None)
    envelopes = getattr(prog, "envelopes", [])
    if name not in pulses or soccfg is None or ch >= len(envelopes):
        return None, None
    pulse = pulses[name]
    params = getattr(pulse, "params", {})
    # Use magnitude so the amplitude plot is non-negative.
    gain = abs(_scalar_value(params.get("gain", 0.0)))
    style = params.get("style", "const")
    try:
        gencfg = soccfg["gens"][ch]
        f_fabric = gencfg["f_fabric"]
        maxv = int(gencfg.get("maxv", 32766))
    except (KeyError, TypeError):
        gencfg = {}
        f_fabric = 1000.0
        maxv = 32766
    scale = maxv if dac_units else 1.0
    if style == "const":
        t = np.array([t_us, t_us, t_us + length_us, t_us + length_us])
        amp = np.array([0.0, gain * scale, gain * scale, 0.0])
        return t, amp
    env_name = params.get("envelope")
    if not env_name or env_name not in envelopes[ch]["envs"]:
        t = np.array([t_us, t_us, t_us + length_us, t_us + length_us])
        amp = np.array([0.0, gain * scale, gain * scale, 0.0])
        return t, amp
    env = envelopes[ch]["envs"][env_name]
    data = np.asarray(env["data"])
    if data.ndim == 2:
        mag = np.sqrt(data[:, 0].astype(float) ** 2 + data[:, 1].astype(float) ** 2)
    else:
        mag = np.abs(data.astype(float))
    if mag.size == 0:
        return None, None
    samps_per_clk = gencfg.get("samps_per_clk", 1)
    n_samps = mag.size
    dt_us = samps_per_clk / f_fabric
    t = t_us + np.arange(n_samps) * dt_us
    max_env = np.max(mag) if np.max(mag) > 0 else 1.0
    amp = (mag / max_env) * gain * scale
    return t, amp


def _get_pulse_unit_envelope_trace(prog, ch: int, name: str, t_us: float, length_us: float):
    """
    Return (t_us_array, env_array) where env_array is unitless 0–1 envelope magnitude.
    This is used to build sweep bands when a pulse parameter (e.g. gain) varies across a loop.
    """
    pulses = getattr(prog, "pulses", {})
    soccfg = getattr(prog, "soccfg", None)
    envelopes = getattr(prog, "envelopes", [])
    if name not in pulses or soccfg is None or ch >= len(envelopes):
        return None, None
    pulse = pulses[name]
    params = getattr(pulse, "params", {})
    style = params.get("style", "const")
    if style == "const":
        t = np.array([t_us, t_us, t_us + length_us, t_us + length_us])
        env = np.array([0.0, 1.0, 1.0, 0.0])
        return t, env

    env_name = params.get("envelope")
    if not env_name or env_name not in envelopes[ch]["envs"]:
        t = np.array([t_us, t_us, t_us + length_us, t_us + length_us])
        env = np.array([0.0, 1.0, 1.0, 0.0])
        return t, env

    envd = envelopes[ch]["envs"][env_name]
    data = np.asarray(envd["data"])
    if data.ndim == 2:
        mag = np.sqrt(data[:, 0].astype(float) ** 2 + data[:, 1].astype(float) ** 2)
    else:
        mag = np.abs(data.astype(float))
    if mag.size == 0:
        return None, None

    try:
        gencfg = soccfg["gens"][ch]
        f_fabric = gencfg["f_fabric"]
        samps_per_clk = gencfg.get("samps_per_clk", 1)
    except (KeyError, TypeError):
        f_fabric = 1000.0
        samps_per_clk = 1
    dt_us = samps_per_clk / f_fabric
    t_core = t_us + np.arange(mag.size) * dt_us
    max_env = float(np.max(mag)) if float(np.max(mag)) > 0 else 1.0
    env_core = (mag / max_env).astype(float)
    # Add explicit edges to baseline for a cleaner filled band.
    t = np.concatenate([[t_us], t_core, [t_us + length_us]])
    env = np.concatenate([[0.0], env_core, [0.0]])
    return t, env


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
):
    """
    Plot a pulse schedule from a compiled QICK asm_v2 program (e.g. AveragerProgramV2).

    Parameters
    ----------
    prog : QickProgramV2
        Compiled program (from pickle or built with soccfg from file).
    ax : matplotlib axes, optional
        If None, current axes or new figure is used.
    max_time_us : float, optional
        Right limit of time axis (us). If None, inferred from schedule.
    gen_ch_labels : dict, optional
        Map gen_ch (int) -> label str for y-axis.
    physical_port_labels : dict, optional
        Map RFDC IDs (e.g. dac '12', adc '20') -> human labels (e.g. 'output 0', 'input 0').
        If provided, these are shown in parentheses after the logical channel (e.g. "ADC 0 (input 0)").
        If not provided, only logical channel is shown (e.g. "ADC 0", "gen 6").
    show_readout_triggers : bool
        If True, show ADC integration windows as separate bars (from Trigger macros).
    show_amplitude : bool
        If True, add a second panel showing input amplitude vs time for each gen channel.
    amplitude_units : str
        "dac" = amplitude in DAC units (0 to maxv, e.g. 32766); "norm" = normalized 0–1.
    amplitude panel
        Uses a linear y-scale.
    title : str, optional
        Plot title.

    Notes
    -----
    - Time axis is in **microseconds** so that relative timing (e.g. qubit pulse length vs
      readout delay) is correct across channels with different clock rates.
    - **Periodic pulses** (e.g. mode="periodic" like spa_pump_cw): plotted as a continuous bar
      from their start time until the next pulse on that same channel (or the end of the plotted
      window if nothing turns them off).
    - **Loops**: only one iteration is visualized (immediate parameter values). Loop-swept
      variables (e.g. qubit pulse power in a gain_loop) and averaging repetition are not
      represented; consider annotating the plot or the program name if that matters.

    Returns
    -------
    ax : matplotlib axes (or tuple of axes if show_amplitude)
    """
    schedule = _extract_schedule(prog)
    if not schedule:
        if ax is None:
            fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No pulse schedule could be extracted from this program.",
                transform=ax.transAxes, ha="center", va="center")
        if title:
            ax.set_title(title)
        return (ax, None) if show_amplitude else ax

    if show_amplitude and ax is None:
        fig, (ax, ax_amp) = plt.subplots(2, 1, figsize=(6, 5), height_ratios=[1.2, 1], sharex=True)
        ax_amp.set_ylabel("Amplitude (DAC units)" if amplitude_units == "dac" else "Amplitude (norm)")
        ax_amp.set_xlabel("Time (µs)")
        ax_amp.grid(True, alpha=0.3)
        if amplitude_units != "dac":
            ax_amp.set_ylim(-0.05, 1.05)
    elif ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax_amp = None

    # Build unique channel list and y-positions
    gen_chs = sorted({s[0] for s in schedule if s[4] == "gen" and s[0] >= 0})
    adc_chs = sorted({s[0] for s in schedule if s[4] == "adc"})
    soccfg = getattr(prog, "soccfg", None)
    ref_ch = gen_chs[0] if gen_chs else 0

    def to_us(t_cy: float, length_cy: float, ch: int, kind: str):
        """Convert (t_cycles, length_cycles) on channel ch to microseconds for a common time axis."""
        if soccfg is None or not hasattr(soccfg, "cycles2us"):
            return float(t_cy) / 1000.0, float(length_cy) / 1000.0
        try:
            if kind == "gen":
                t_us = soccfg.cycles2us(t_cy, gen_ch=ch)
                len_us = soccfg.cycles2us(length_cy, gen_ch=ch)
            else:
                t_us = soccfg.cycles2us(t_cy, ro_ch=ch)
                len_us = soccfg.cycles2us(length_cy, ro_ch=ch)
            return t_us, len_us
        except Exception:
            return float(t_cy) / 1000.0, float(length_cy) / 1000.0

    y_pos = {}
    idx = 0
    for ch in gen_chs:
        y_pos[("gen", ch)] = idx
        idx += 1
    if show_readout_triggers and adc_chs:
        for ch in adc_chs:
            y_pos[("adc", ch)] = idx
            idx += 1

    pulses = getattr(prog, "pulses", {})

    def _is_periodic_pulse(pulse_name: str) -> bool:
        p = pulses.get(pulse_name)
        params = getattr(p, "params", {}) if p is not None else {}
        return params.get("mode") == "periodic"

    def _gain_sweep_range(pulse_name: str):
        """Return (gmin, gmax) if gain varies across a loop, else None."""
        if not hasattr(prog, "get_pulse_param"):
            return None
        try:
            g = prog.get_pulse_param(pulse_name, "gain", as_array=True)
        except Exception:
            return None
        arr = np.asarray(g).astype(float).ravel()
        if arr.size < 2:
            return None
        gmin = float(np.min(arr))
        gmax = float(np.max(arr))
        if np.isclose(gmin, gmax, rtol=0, atol=1e-6):
            return None
        return gmin, gmax

    def _has_param_sweep(pulse_name: str, param: str) -> bool:
        """Return True if pulse param varies across a loop (array has >1 unique value)."""
        if not hasattr(prog, "get_pulse_param"):
            return False
        try:
            v = prog.get_pulse_param(pulse_name, param, as_array=True)
        except Exception:
            return False
        arr = np.asarray(v).astype(float).ravel()
        if arr.size < 2:
            return False
        return not np.isclose(float(np.min(arr)), float(np.max(arr)), rtol=0, atol=1e-9)

    # Plot gen pulses as horizontal bars with pulse name on each block (time axis in µs)
    ref_ends_us = []
    for s in schedule:
        ch, kind = s[0], s[4]
        t_us, len_us = to_us(s[2], s[3], ch, kind)
        ref_ends_us.append(t_us + len_us)
    end_us = max(ref_ends_us, default=1.0)
    end_us_one_shot = float(end_us)

    # If the program has an explicit sweep loop (common key: cfg["steps"]), annotate it.
    cfg_obj = getattr(prog, "cfg", {}) or {}
    try:
        loop_steps = int(cfg_obj.get("steps", 1))
    except Exception:
        loop_steps = 1
    loop_steps = max(1, loop_steps)

    # For mode="periodic" pulses, extend until next pulse on same channel (or end_us).
    # This reflects continuous playback until explicitly changed/turned off.
    gen_events = []
    for ch, name, t_cy, length_cy, kind in schedule:
        if kind != "gen" or ch < 0:
            continue
        t_us, length_us = to_us(t_cy, length_cy, ch, kind)
        gen_events.append((int(ch), str(name), float(t_us), float(length_us)))
    gen_events_by_ch = {}
    for ch, name, t_us, length_us in gen_events:
        gen_events_by_ch.setdefault(ch, []).append((name, t_us, length_us))
    periodic_draw_len_us = {}  # (ch, name, t_us) -> draw_length_us
    for ch, events in gen_events_by_ch.items():
        events_sorted = sorted(events, key=lambda x: x[1])
        for i, (name, t_us, length_us) in enumerate(events_sorted):
            if not _is_periodic_pulse(name):
                continue
            # Periodic pulses should extend until the next *later* event on the same channel.
            # Some programs schedule a "turnoff/off" pulse at the same timestamp (e.g. via _cleanup),
            # which would otherwise make the periodic draw length zero.
            k = i + 1
            while k < len(events_sorted) and events_sorted[k][1] <= t_us + 1e-12:
                k += 1
            next_start = events_sorted[k][1] if k < len(events_sorted) else end_us
            draw_len = max(0.0, next_start - t_us)
            periodic_draw_len_us[(ch, name, t_us)] = draw_len

    # We keep the x-axis to a single-shot window. If a sweep loop is present, we annotate it
    # (repetitions are real, but drawing them literally would make the plot unreadable).
    end_us = end_us_one_shot

    # For readability: if multiple pulses land at the same (ch, t_us), drop the "turnoff"-like
    # pulse when a periodic pulse exists at the same timestamp (common for _cleanup artifacts).
    same_time_by_ch = {}
    for ch, name, t_us, length_us in gen_events:
        key = (int(ch), float(t_us))
        same_time_by_ch.setdefault(key, []).append(str(name))
    skip_gen_at_time = set()
    for (ch, t_us), names in same_time_by_ch.items():
        if len(names) < 2:
            continue
        if any(_is_periodic_pulse(n) for n in names):
            for n in names:
                if (not _is_periodic_pulse(n)) and ("turnoff" in n.lower() or "off" in n.lower()):
                    skip_gen_at_time.add((ch, n, t_us))

    # (Loop annotation removed for plot cleanliness.)

    # Draw true durations (no minimum-width inflation).
    min_bar_width_us = 0.0
    for ch, name, t_cy, length_cy, kind in schedule:
        if kind != "gen" or ch < 0:
            continue
        t_us, length_us = to_us(t_cy, length_cy, ch, kind)
        y = y_pos.get(("gen", ch), 0)
        color = plt.cm.tab10(ch % 10)
        is_periodic = _is_periodic_pulse(name)
        sweep = _gain_sweep_range(name)
        if (int(ch), str(name), float(t_us)) in skip_gen_at_time:
            continue
        draw_len = periodic_draw_len_us.get((int(ch), str(name), float(t_us))) if is_periodic else None
        if draw_len is None:
            draw_len = length_us
        width_us = max(draw_len, min_bar_width_us)
        ax.barh(
            y,
            width_us,
            left=t_us,
            height=0.7,
            align="center",
            color=color,
            edgecolor="black",
            linewidth=0.5,
            hatch="////" if is_periodic else None,
            alpha=0.55 if is_periodic else 1.0,
        )

    # Plot ADC integration windows as separate shapes (when available)
    adc_green = "#1a7a1a"  # Darker green so integration is clearly visible
    if show_readout_triggers:
        for ch, name, t_cy, length_cy, kind in schedule:
            if kind != "adc":
                continue
            y = y_pos.get(("adc", ch))
            if y is None:
                continue
            t_us, length_us = to_us(t_cy, length_cy, ch, kind)
            width_us = max(length_us, 0.01)
            ax.barh(
                y,
                width_us,
                left=t_us,
                height=0.45,
                align="center",
                color=adc_green,
                alpha=0.65,
                edgecolor="black",
                linewidth=1.0,
            )

    # Y-axis labels: show which DAC/ADC IDs are used; allow overriding with a human mapping.
    y_ticks = []
    y_labels = []
    for ch in gen_chs:
        y_ticks.append(y_pos[("gen", ch)])
        label = (gen_ch_labels or {}).get(ch, "gen %d" % ch)
        if soccfg is not None:
            try:
                dac_id = soccfg["gens"][ch].get("dac")
            except Exception:
                dac_id = None
            if dac_id is not None:
                dac_key = str(dac_id)
                phys = physical_port_labels.get(dac_key) if physical_port_labels else None
                label = f"{label} ({phys or ('dac ' + dac_key)})"
        y_labels.append(label)
    if show_readout_triggers:
        for ch in adc_chs:
            y_ticks.append(y_pos[("adc", ch)])
            label = "ro %d" % ch
            if soccfg is not None:
                try:
                    adc_id = soccfg["readouts"][ch].get("adc")
                except Exception:
                    adc_id = None
                if adc_id is not None:
                    adc_key = str(adc_id)
                    phys = physical_port_labels.get(adc_key) if physical_port_labels else None
                    label = f"{label} ({phys or ('adc ' + adc_key)})"
            y_labels.append(label)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylim(-0.5, len(y_pos) - 0.5)
    if max_time_us is not None:
        end_us = max_time_us
        ax.set_xlim(0, end_us)
    else:
        gen_ends_us = []
        for s in schedule:
            if s[4] != "gen" or s[0] < 0:
                continue
            t_u, l_u = to_us(s[2], s[3], s[0], "gen")
            gen_ends_us.append(t_u + l_u)
        adc_ends_us = []
        for s in schedule:
            if s[4] != "adc":
                continue
            t_u, l_u = to_us(s[2], s[3], s[0], "adc")
            adc_ends_us.append(t_u + l_u)
        all_ends = sorted(gen_ends_us + adc_ends_us)
        if all_ends:
            full_end = all_ends[-1]
            p80_idx = min(int(0.8 * len(all_ends)), len(all_ends) - 1)
            p80 = all_ends[p80_idx]
            if full_end > p80 * 1.5:
                end_us = p80 * 1.15
            else:
                end_us = full_end * 1.01
        else:
            end_us = max(ref_ends_us, default=1.0) * 1.01
        # Add small padding so final events aren't flush to the right edge.
        end_us = end_us * 1.02
        ax.set_xlim(0, end_us)
    ax.set_xlabel("Time (µs)")
    if title:
        ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)

    # Amplitude vs time panel (x-axis in µs)
    if ax_amp is not None:
        dac_units = amplitude_units == "dac"
        gen_pulses = [(s[0], s[1], s[2], s[3]) for s in schedule if s[4] == "gen" and s[0] >= 0]
        # Darker, more legible colors and thicker lines for the amplitude plot
        colors = plt.cm.tab10
        linewidth = 2.0
        legend_channels = set()
        # For visualization only: enforce a minimum visible width for very short const pulses
        # so they don't look like delta spikes on long time windows.
        min_vis_us = max(0.01, end_us * 0.002)
        for ch, name, t_cy, length_cy in gen_pulses:
            t_us, length_us = to_us(t_cy, length_cy, ch, "gen")
            # Extend periodic const pulses so amplitude matches the continuous schedule bar.
            if _is_periodic_pulse(name):
                ext = periodic_draw_len_us.get((int(ch), str(name), float(t_us)))
                if ext is not None and ext > 0:
                    length_us = ext
            sweep = _gain_sweep_range(name)
            if sweep is not None:
                env_t, env = _get_pulse_unit_envelope_trace(prog, ch, name, t_us, length_us)
                if env_t is not None and env is not None:
                    try:
                        maxv = int(soccfg["gens"][ch].get("maxv", 32766)) if (dac_units and soccfg is not None) else 1
                    except Exception:
                        maxv = 32766 if dac_units else 1
                    gmin, gmax = sweep
                    # Use magnitude so the amplitude plot is non-negative.
                    gmin, gmax = abs(gmin), abs(gmax)
                    amp_lo = env * min(gmin, gmax) * maxv
                    amp_hi = env * max(gmin, gmax) * maxv
                    label = (gen_ch_labels or {}).get(ch, "gen %d" % ch) if ch not in legend_channels else "_nolegend_"
                    if ch not in legend_channels:
                        legend_channels.add(ch)
                    ax_amp.fill_between(env_t, amp_lo, amp_hi, color=colors(ch % 10), alpha=0.35, label="_nolegend_", linewidth=0)
                    # Median line gets the legend entry so legend color matches the trace
                    ax_amp.plot(env_t, (amp_lo + amp_hi) / 2, color=colors(ch % 10), linewidth=linewidth, zorder=2, label=label)
            else:
                t_arr, amp_arr = _get_pulse_amplitude_trace(prog, ch, name, t_us, length_us, dac_units=dac_units)
                if t_arr is not None and amp_arr is not None:
                    label = (gen_ch_labels or {}).get(ch, "gen %d" % ch) if ch not in legend_channels else "_nolegend_"
                    if ch not in legend_channels:
                        legend_channels.add(ch)
                    # If this is a const pulse segment (2 points), widen only for display if needed.
                    if t_arr.size == 2:
                        w = float(t_arr[1] - t_arr[0])
                        if w < min_vis_us:
                            t_arr = np.array([t_arr[0], t_arr[0] + min_vis_us])
                    ax_amp.plot(t_arr, amp_arr, color=colors(ch % 10), label=label, linewidth=linewidth)
        # Shade readout integration windows for context (same green as top panel)
        if show_readout_triggers:
            for ch, name, t_cy, length_cy, kind in schedule:
                if kind != "adc":
                    continue
                t_us, length_us = to_us(t_cy, length_cy, ch, kind)
                ax_amp.axvspan(t_us, t_us + length_us, color=adc_green, alpha=0.25, lw=0)
        if gen_pulses or (show_readout_triggers and adc_chs):
            handles, labels = ax_amp.get_legend_handles_labels()
            if show_readout_triggers and adc_chs:
                handles.append(Patch(facecolor=adc_green, alpha=0.25, edgecolor="black"))
                labels.append("integration" if len(adc_chs) > 1 else "ro %d (integration)" % adc_chs[0])
            ax_amp.legend(
                handles=handles,
                labels=labels,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                fontsize=6,
                framealpha=0.9,
            )
            # Make room on the right for the outside legend.
            try:
                ax_amp.figure.subplots_adjust(right=0.78)
            except Exception:
                pass
        ax_amp.set_xlim(0, end_us)
        ax_amp.set_ylim(bottom=0)

    return (ax, ax_amp) if ax_amp is not None else ax


def visualize_from_pickle(
    pickle_path: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Load a compiled program from a pickle file (e.g. compiled_program_pickle from a measurement)
    and plot its pulse schedule. No RFSoC connection required.

    Parameters
    ----------
    pickle_path : str
        Path to the .pkl file containing the compiled program.
    output_path : str, optional
        If set, save the figure to this path.
    title : str, optional
        Plot title.
    """
    import cloudpickle
    with open(pickle_path, "rb") as f:
        prog = cloudpickle.load(f)
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_pulse_schedule(prog, ax=ax, title=title or "Pulse schedule")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print("Saved figure to %s" % output_path)
    plt.show()
    return prog, ax


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pulse_visualizer.py <path_to_compiled_program.pkl> [output_figure.png]")
        print("Or from Python:")
        print("  from qcvt import load_soccfg_from_json, plot_pulse_schedule, save_soccfg_to_json")
        print("  # When connected once: save_soccfg_to_json(soc, 'qick_config.json')")
        print("  soccfg = load_soccfg_from_json('qick_config.json')")
        print("  prog = YourProgram(soccfg, reps=2, cfg=config)")
        print("  plot_pulse_schedule(prog); plt.show()")
        sys.exit(0)
    pkl_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    visualize_from_pickle(pkl_path, output_path=out_path)


def export_amplitude_traces_csv(
    prog,
    csv_path: str,
    t0_us: float,
    t1_us: float,
    amplitude_units: str = "dac",
) -> str:
    """
    Export raw amplitude traces for all gen channels used in a program.

    The CSV is sampled on the union of all pulse edge times within [t0_us, t1_us],
    so piecewise-constant pulses are represented exactly (no arbitrary resampling).

    Columns:
      - time_us
      - one column per gen channel label: gen <ch>
    """
    if amplitude_units not in ("dac", "norm"):
        raise ValueError("amplitude_units must be 'dac' or 'norm'")

    schedule = _extract_schedule(prog)
    if not schedule:
        raise RuntimeError("No schedule could be extracted from this program.")

    soccfg = getattr(prog, "soccfg", None)
    pulses = getattr(prog, "pulses", {}) or {}

    def is_periodic(pulse_name: str) -> bool:
        p = pulses.get(pulse_name)
        params = getattr(p, "params", {}) if p is not None else {}
        return params.get("mode") == "periodic"

    def to_us(t_cy: float, length_cy: float, ch: int, kind: str):
        if soccfg is None or not hasattr(soccfg, "cycles2us"):
            return float(t_cy) / 1000.0, float(length_cy) / 1000.0
        if kind == "gen":
            return soccfg.cycles2us(t_cy, gen_ch=ch), soccfg.cycles2us(length_cy, gen_ch=ch)
        return soccfg.cycles2us(t_cy, ro_ch=ch), soccfg.cycles2us(length_cy, ro_ch=ch)

    gen_events = [(s[0], s[1], s[2], s[3], s[4]) for s in schedule if s[4] == "gen" and s[0] >= 0]
    gen_chs = sorted({int(ch) for ch, *_rest in gen_events})
    if not gen_chs:
        raise RuntimeError("No generator pulses found in schedule.")

    # Collect edge times from amplitude traces (within window) for exact sampling.
    edge_times = {float(t0_us), float(t1_us)}
    traces = {ch: [] for ch in gen_chs}  # list of (t_arr, amp_arr)
    dac_units = amplitude_units == "dac"
    for ch, name, t_cy, length_cy, kind in gen_events:
        t_us, length_us = to_us(t_cy, length_cy, int(ch), "gen")
        t_arr, amp_arr = _get_pulse_amplitude_trace(prog, int(ch), str(name), float(t_us), float(length_us), dac_units=dac_units)
        if t_arr is None or amp_arr is None:
            continue
        t_arr = np.asarray(t_arr, dtype=float)
        amp_arr = np.asarray(amp_arr, dtype=float)
        # Clip edges for sampling set
        for t in t_arr:
            if t0_us <= float(t) <= t1_us:
                edge_times.add(float(t))
        traces[int(ch)].append((t_arr, amp_arr))

    times = np.array(sorted(edge_times), dtype=float)

    # Evaluate piecewise traces at each time.
    # For each channel, amplitude is the max of overlapping pulses (matches visual stacking expectation).
    amp_mat = np.zeros((times.size, len(gen_chs)), dtype=float)
    for j, ch in enumerate(gen_chs):
        segs = traces.get(ch, [])
        if not segs:
            continue
        a = np.zeros(times.size, dtype=float)
        for t_arr, amp_arr in segs:
            # Interpret 4-point box traces as: [t0,t0,t1,t1] with [0,amp,amp,0]
            if t_arr.size == 4 and amp_arr.size == 4:
                t0, t1 = float(t_arr[1]), float(t_arr[2])
                amp = float(amp_arr[1])
                mask = (times >= t0) & (times <= t1)
                a[mask] = np.maximum(a[mask], amp)
            else:
                # Generic interpolation (envelopes): hold last value (zero-order hold)
                order = np.argsort(t_arr)
                tt = t_arr[order]
                aa = amp_arr[order]
                # For each time, take value at last sample <= time
                idx = np.searchsorted(tt, times, side="right") - 1
                idx = np.clip(idx, 0, len(tt) - 1)
                vals = aa[idx]
                # Zero outside the trace time window
                vals[(times < tt[0]) | (times > tt[-1])] = 0.0
                a = np.maximum(a, vals)
        amp_mat[:, j] = a

    # Write CSV
    import csv as _csv

    with open(csv_path, "w", newline="") as f:
        w = _csv.writer(f)
        header = ["time_us"] + [f"gen_{ch}" for ch in gen_chs]
        w.writerow(header)
        for i, t in enumerate(times):
            row = [f"{t:.9f}"] + [f"{amp_mat[i, j]:.9f}" for j in range(len(gen_chs))]
            w.writerow(row)

    # Also save NPZ alongside for lossless arrays
    npz_path = csv_path.rsplit(".", 1)[0] + ".npz"
    np.savez(npz_path, time_us=times, gen_chs=np.array(gen_chs, dtype=int), amp=amp_mat)
    return npz_path


def export_edge_matrices_csv_legacy(
    prog,
    out_prefix: str,
    lanes: List[dict],
    t0_us: float = 0.0,
    t1_us: Optional[float] = None,
    gap_threshold_us: Optional[float] = None,
    amplitude_units: str = "dac",
) -> Tuple[str, str]:
    """
    Build two "edge matrices" as CSVs:

    1) State-only: entries are "on"/"off"
    2) State+amplitude: entries are "on (<amp>)"/"off"

    Only rising/falling edges are included as timestamp columns. If a long time passes with no
    state changes, an extra column with timestamp "-" is inserted, showing the current state.

    lanes: list of dicts like:
      {"label": "pulse", "kind": "gen", "ch": 6}
      {"label": "ro 0", "kind": "adc", "ch": 0}
    """
    if amplitude_units not in ("dac", "norm"):
        raise ValueError("amplitude_units must be 'dac' or 'norm'")

    schedule = _extract_schedule(prog)
    if not schedule:
        raise RuntimeError("No schedule could be extracted from this program.")

    soccfg = getattr(prog, "soccfg", None)
    pulses = getattr(prog, "pulses", {}) or {}

    def is_periodic(pulse_name: str) -> bool:
        p = pulses.get(pulse_name)
        params = getattr(p, "params", {}) if p is not None else {}
        return params.get("mode") == "periodic"
    pulses = getattr(prog, "pulses", {})

    def to_us(t_cy: float, length_cy: float, ch: int, kind: str):
        if soccfg is None or not hasattr(soccfg, "cycles2us"):
            return float(t_cy) / 1000.0, float(length_cy) / 1000.0
        if kind == "gen":
            return soccfg.cycles2us(t_cy, gen_ch=ch), soccfg.cycles2us(length_cy, gen_ch=ch)
        return soccfg.cycles2us(t_cy, ro_ch=ch), soccfg.cycles2us(length_cy, ro_ch=ch)

    def is_periodic(pulse_name: str) -> bool:
        p = pulses.get(pulse_name)
        params = getattr(p, "params", {}) if p is not None else {}
        return params.get("mode") == "periodic"

    # Compute a plot window end if not provided.
    all_ends = []
    for ch, name, t_cy, length_cy, kind in schedule:
        t_us, length_us = to_us(t_cy, length_cy, ch, kind)
        all_ends.append(float(t_us + length_us))
    if t1_us is None:
        t1_us = max(all_ends, default=float(t0_us))

    if gap_threshold_us is None:
        span = max(0.0, float(t1_us) - float(t0_us))
        gap_threshold_us = max(1.0, 0.05 * span)

    dac_units = amplitude_units == "dac"

    # Build gen events with periodic extension (same rule as plotter: extend until next later event).
    gen_events = []
    for ch, name, t_cy, length_cy, kind in schedule:
        if kind != "gen" or ch < 0:
            continue
        t_us, length_us = to_us(t_cy, length_cy, int(ch), "gen")
        gen_events.append((int(ch), str(name), float(t_us), float(length_us)))
    gen_events_by_ch = {}
    for ch, name, t_us, length_us in gen_events:
        gen_events_by_ch.setdefault(ch, []).append((name, t_us, length_us))

    periodic_len = {}  # (ch, name, t_us) -> draw_len
    for ch, events in gen_events_by_ch.items():
        events_sorted = sorted(events, key=lambda x: x[1])
        for i, (nm, t_us, length_us) in enumerate(events_sorted):
            if not is_periodic(nm):
                continue
            k = i + 1
            while k < len(events_sorted) and events_sorted[k][1] <= t_us + 1e-12:
                k += 1
            next_start = events_sorted[k][1] if k < len(events_sorted) else float(t1_us)
            periodic_len[(ch, nm, t_us)] = max(0.0, next_start - t_us)

    # For each lane, build a list of (t_on, t_off, amp) intervals.
    lane_intervals = {ln["label"]: [] for ln in lanes}
    for ln in lanes:
        label = ln["label"]
        kind = ln["kind"]
        ch = int(ln["ch"])
        if kind == "adc":
            for c, name, t_cy, length_cy, knd in schedule:
                if knd != "adc" or int(c) != ch:
                    continue
                t_us, length_us = to_us(t_cy, length_cy, ch, "adc")
                lane_intervals[label].append((float(t_us), float(t_us + length_us), 1.0))
        else:
            for c, name, t_cy, length_cy, knd in schedule:
                if knd != "gen" or int(c) != ch:
                    continue
                t_us, length_us = to_us(t_cy, length_cy, ch, "gen")
                if is_periodic(name):
                    ext = periodic_len.get((ch, str(name), float(t_us)))
                    if ext is not None and ext > 0:
                        length_us = ext
                t_arr, amp_arr = _get_pulse_amplitude_trace(prog, ch, str(name), float(t_us), float(length_us), dac_units=dac_units)
                amp = float(np.nanmax(amp_arr)) if amp_arr is not None else 0.0
                lane_intervals[label].append((float(t_us), float(t_us + length_us), amp))

    # Collect all edge times (on/off) across lanes within window.
    edge_times = {float(t0_us), float(t1_us)}
    for label, intervals in lane_intervals.items():
        for a, b, amp in intervals:
            if b <= t0_us or a >= t1_us:
                continue
            edge_times.add(max(float(t0_us), a))
            edge_times.add(min(float(t1_us), b))
    edge_times = sorted(edge_times)

    def state_at(label: str, t: float) -> Tuple[str, float]:
        """Return ('on'/'off', amp) for a lane at time t (right-continuous)."""
        st = "off"
        amp = 0.0
        for a, b, ap in lane_intervals[label]:
            if a <= t < b:
                st = "on"
                amp = max(amp, float(ap))
        return st, amp

    # Build the matrix "columns" (as rows in CSV): timestamp + per-lane value.
    col_spec = []  # list of (timestamp_str, snapshot_dict)
    prev_t = None
    prev_snapshot = None
    for t in edge_times:
        if prev_t is not None and (t - prev_t) > gap_threshold_us:
            # Insert a spacer column with "-" and the state in the middle of the steady region.
            mid = (prev_t + t) / 2
            snap = {ln["label"]: state_at(ln["label"], mid) for ln in lanes}
            col_spec.append(("-", snap))
        snap = {ln["label"]: state_at(ln["label"], t + 1e-15) for ln in lanes}
        col_spec.append((f"{t:.9f}", snap))
        prev_t = t

    state_csv = out_prefix + "_edges_state.csv"
    amp_csv = out_prefix + "_edges_amp.csv"

    import csv as _csv
    lane_labels = [ln["label"] for ln in lanes]

    with open(state_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["timestamp_us"] + lane_labels)
        for ts, snap in col_spec:
            row = [ts] + [snap[lbl][0] for lbl in lane_labels]
            w.writerow(row)

    with open(amp_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["timestamp_us"] + lane_labels)
        for ts, snap in col_spec:
            out = [ts]
            for lbl in lane_labels:
                st, ap = snap[lbl]
                if st == "off":
                    out.append("off")
                else:
                    out.append(f"on ({ap:.6g})")
            w.writerow(out)

    return state_csv, amp_csv


def export_edge_matrices_csv_old(
    prog,
    out_prefix: str,
    t0_us: float,
    t1_us: Optional[float],
    rows: Optional[List[Tuple[str, str, int]]] = None,
    gap_threshold_us: Optional[float] = None,
    amplitude_units: str = "dac",
) -> Tuple[str, str]:
    """
    Export two "edge matrices" which only log rising/falling edges.

    Matrix 1 (state): entries are "on"/"off".
    Matrix 2 (amplitude): entries are "on <amp>"/"off", where <amp> is the max amplitude at that edge time.

    Columns are edge timestamps in microseconds. If there is a long period with no changes between
    consecutive edge timestamps, a column with timestamp "-" is inserted between them, containing the
    current state (and amplitude for matrix 2).

    Parameters
    ----------
    out_prefix:
        Output path prefix (without extension). Writes:
          - f"{out_prefix}_state.csv"
          - f"{out_prefix}_amp.csv"
    rows:
        List of (row_label, kind, ch). kind is "gen" or "adc".
        Default (if None): all gen channels labeled "gen <ch>" plus any adc channels labeled "ro <ch>".
    """
    if amplitude_units not in ("dac", "norm"):
        raise ValueError("amplitude_units must be 'dac' or 'norm'")

    schedule = _extract_schedule(prog)
    if not schedule:
        raise RuntimeError("No schedule could be extracted from this program.")

    soccfg = getattr(prog, "soccfg", None)

    def to_us(t_cy: float, length_cy: float, ch: int, kind: str):
        if soccfg is None or not hasattr(soccfg, "cycles2us"):
            return float(t_cy) / 1000.0, float(length_cy) / 1000.0
        if kind == "gen":
            return soccfg.cycles2us(t_cy, gen_ch=ch), soccfg.cycles2us(length_cy, gen_ch=ch)
        return soccfg.cycles2us(t_cy, ro_ch=ch), soccfg.cycles2us(length_cy, ro_ch=ch)

    gen_chs = sorted({int(s[0]) for s in schedule if s[4] == "gen" and s[0] >= 0})
    adc_chs = sorted({int(s[0]) for s in schedule if s[4] == "adc"})
    if rows is None:
        rows = [(f"gen {ch}", "gen", ch) for ch in gen_chs] + [(f"ro {ch}", "adc", ch) for ch in adc_chs]

    intervals = {}  # (kind,ch) -> list of (t0,t1,amp_max)
    dac_units = amplitude_units == "dac"

    for ch, name, t_cy, length_cy, kind in schedule:
        if kind != "gen" or int(ch) < 0:
            continue
        t_us, len_us = to_us(t_cy, length_cy, int(ch), "gen")
        t_arr, amp_arr = _get_pulse_amplitude_trace(prog, int(ch), str(name), float(t_us), float(len_us), dac_units=dac_units)
        if t_arr is None or amp_arr is None:
            continue
        t_arr = np.asarray(t_arr, dtype=float)
        amp_arr = np.asarray(amp_arr, dtype=float)
        if t_arr.size == 4 and amp_arr.size == 4:
            a = float(amp_arr[1])
            if a != 0:
                intervals.setdefault(("gen", int(ch)), []).append((float(t_arr[1]), float(t_arr[2]), a))
        else:
            a = float(np.nanmax(amp_arr)) if amp_arr.size else 0.0
            if a != 0:
                intervals.setdefault(("gen", int(ch)), []).append((float(np.nanmin(t_arr)), float(np.nanmax(t_arr)), a))

    for ch, _name, t_cy, length_cy, kind in schedule:
        if kind != "adc":
            continue
        t_us, len_us = to_us(t_cy, length_cy, int(ch), "adc")
        intervals.setdefault(("adc", int(ch)), []).append((float(t_us), float(t_us + len_us), 1.0))

    def state_amp_at(kind: str, ch: int, t: float) -> Tuple[bool, float]:
        segs = intervals.get((kind, ch), [])
        if not segs:
            return False, 0.0
        a = 0.0
        on = False
        for t0, t1, amp in segs:
            # Half-open intervals: [t0, t1). This makes the end timestamp represent the falling edge.
            if t0 <= t < t1:
                on = True
                a = max(a, float(amp))
        return on, a

    edge_times = {float(t0_us), float(t1_us)}
    for segs in intervals.values():
        for a, b, _amp in segs:
            if t0_us <= a <= t1_us:
                edge_times.add(float(a))
            if t0_us <= b <= t1_us:
                edge_times.add(float(b))
    edge_times = sorted(edge_times)

    if gap_threshold_us is None:
        gap_threshold_us = max(0.5, min(5.0, 0.10 * max(0.0, float(t1_us - t0_us))))

    columns: List[Any] = []
    for i, t in enumerate(edge_times):
        columns.append(float(t))
        if i + 1 < len(edge_times):
            dt = float(edge_times[i + 1] - t)
            if dt > float(gap_threshold_us):
                columns.append("-")

    state_rows = []
    amp_rows = []
    for label, kind, ch in rows:
        srow = [label]
        arow = [label]
        current_on = None
        current_amp = None
        for col in columns:
            if col == "-":
                on = bool(current_on) if current_on is not None else False
                amp = float(current_amp) if current_amp is not None else 0.0
            else:
                on, amp = state_amp_at(kind, int(ch), float(col))
                current_on, current_amp = on, amp
            srow.append("on" if on else "off")
            arow.append(f"on {amp:.6g}" if on else "off")
        state_rows.append(srow)
        amp_rows.append(arow)

    import csv as _csv
    state_path = f"{out_prefix}_state.csv"
    amp_path = f"{out_prefix}_amp.csv"
    header = ["row"] + [("-" if c == "-" else f"{float(c):.9f}") for c in columns]

    with open(state_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(header)
        w.writerows(state_rows)

    with open(amp_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(header)
        w.writerows(amp_rows)

    return state_path, amp_path


def export_edge_matrices_csv(
    prog,
    out_prefix: str,
    t0_us: float,
    t1_us: Optional[float],
    rows: Optional[List[Tuple[str, str, int]]] = None,
    gap_threshold_us: Optional[float] = None,
    amplitude_units: str = "dac",
) -> Tuple[str, str]:
    """
    Export two "edge matrices" which only log rising/falling edges.

    Matrix 1 (state): entries are "on"/"off".
    Matrix 2 (amplitude): entries are "on <amp>"/"off", where <amp> is the max amplitude at that edge time.

    Columns are edge timestamps in microseconds. If there is a long period with no changes between
    consecutive edge timestamps, a column with timestamp "-" is inserted between them, containing the
    current state (and amplitude for matrix 2).

    Parameters
    ----------
    out_prefix:
        Output path prefix (without extension). Writes:
          - f"{out_prefix}_state.csv"
          - f"{out_prefix}_amp.csv"
    rows:
        List of (row_label, kind, ch). kind is "gen" or "adc".
        Default (if None): all gen channels labeled "gen <ch>" plus any adc channels labeled "ro <ch>".
    """
    if amplitude_units not in ("dac", "norm"):
        raise ValueError("amplitude_units must be 'dac' or 'norm'")

    schedule = _extract_schedule(prog)
    if not schedule:
        raise RuntimeError("No schedule could be extracted from this program.")

    soccfg = getattr(prog, "soccfg", None)
    pulses = getattr(prog, "pulses", {}) or {}

    def is_periodic(pulse_name: str) -> bool:
        p = pulses.get(pulse_name)
        params = getattr(p, "params", {}) if p is not None else {}
        return params.get("mode") == "periodic"

    def to_us(t_cy: float, length_cy: float, ch: int, kind: str):
        if soccfg is None or not hasattr(soccfg, "cycles2us"):
            return float(t_cy) / 1000.0, float(length_cy) / 1000.0
        if kind == "gen":
            return soccfg.cycles2us(t_cy, gen_ch=ch), soccfg.cycles2us(length_cy, gen_ch=ch)
        return soccfg.cycles2us(t_cy, ro_ch=ch), soccfg.cycles2us(length_cy, ro_ch=ch)

    # Collect gen/adc channels in schedule.
    gen_chs = sorted({int(s[0]) for s in schedule if s[4] == "gen" and s[0] >= 0})
    adc_chs = sorted({int(s[0]) for s in schedule if s[4] == "adc"})
    if rows is None:
        rows = [(f"gen {ch}", "gen", ch) for ch in gen_chs] + [(f"ro {ch}", "adc", ch) for ch in adc_chs]

    # Build intervals per row.
    intervals = {}  # (kind,ch) -> list of (t0,t1,amp_max)
    dac_units = amplitude_units == "dac"

    # If t1_us not provided, infer from the raw schedule ends (before periodic extension).
    if t1_us is None:
        ends = []
        for ch, _name, t_cy, length_cy, kind in schedule:
            t_u, l_u = to_us(t_cy, length_cy, int(ch), "gen" if kind == "gen" else "adc")
            ends.append(float(t_u + l_u))
        t1_us = max(ends, default=float(t0_us))

    # Precompute periodic extensions: periodic pulses extend until the next *later* event
    # on the same channel (mirrors plotter behavior).
    gen_events = []
    for ch, name, t_cy, length_cy, kind in schedule:
        if kind != "gen" or int(ch) < 0:
            continue
        t_us, len_us = to_us(t_cy, length_cy, int(ch), "gen")
        gen_events.append((int(ch), str(name), float(t_us), float(len_us)))
    gen_events_by_ch = {}
    for ch, name, t_us, len_us in gen_events:
        gen_events_by_ch.setdefault(ch, []).append((name, t_us, len_us))

    periodic_len_us = {}  # (ch, name, t_us) -> draw_len_us
    for ch, events in gen_events_by_ch.items():
        events_sorted = sorted(events, key=lambda x: x[1])
        for i, (nm, t_us, len_us) in enumerate(events_sorted):
            if not is_periodic(nm):
                continue
            k = i + 1
            while k < len(events_sorted) and events_sorted[k][1] <= t_us + 1e-12:
                k += 1
            next_start = events_sorted[k][1] if k < len(events_sorted) else None
            if next_start is None:
                # If no later event exists, extend to the end of the window (filled in later).
                periodic_len_us[(ch, nm, t_us)] = max(0.0, float(t1_us) - float(t_us))
            else:
                periodic_len_us[(ch, nm, t_us)] = max(0.0, float(next_start) - float(t_us))

    # Gen intervals: use amplitude trace generator; for const pulses this is exact.
    for ch, name, t_cy, length_cy, kind in schedule:
        if kind != "gen" or int(ch) < 0:
            continue
        t_us, len_us = to_us(t_cy, length_cy, int(ch), "gen")
        if is_periodic(str(name)):
            ext = periodic_len_us.get((int(ch), str(name), float(t_us)))
            if ext is not None and ext > 0:
                len_us = float(ext)
        t_arr, amp_arr = _get_pulse_amplitude_trace(prog, int(ch), str(name), float(t_us), float(len_us), dac_units=dac_units)
        if t_arr is None or amp_arr is None:
            continue
        t_arr = np.asarray(t_arr, dtype=float)
        amp_arr = np.asarray(amp_arr, dtype=float)
        if t_arr.size == 4 and amp_arr.size == 4:
            a = float(amp_arr[1])
            if a != 0:
                intervals.setdefault(("gen", int(ch)), []).append((float(t_arr[1]), float(t_arr[2]), a))
        else:
            # Envelope: approximate interval as [min(t), max(t)] with peak amplitude.
            a = float(np.nanmax(amp_arr)) if amp_arr.size else 0.0
            if a != 0:
                intervals.setdefault(("gen", int(ch)), []).append((float(np.nanmin(t_arr)), float(np.nanmax(t_arr)), a))

    # ADC integration intervals from Trigger macro windows.
    for ch, _name, t_cy, length_cy, kind in schedule:
        if kind != "adc":
            continue
        t_us, len_us = to_us(t_cy, length_cy, int(ch), "adc")
        intervals.setdefault(("adc", int(ch)), []).append((float(t_us), float(t_us + len_us), 1.0))

    # Helper: compute state and amplitude at time t.
    def state_amp_at(kind: str, ch: int, t: float) -> Tuple[bool, float]:
        segs = intervals.get((kind, ch), [])
        if not segs:
            return False, 0.0
        a = 0.0
        on = False
        for t0, t1, amp in segs:
            # Half-open intervals: [t0, t1). This makes the end timestamp represent the falling edge.
            if t0 <= t < t1:
                on = True
                a = max(a, float(amp))
        return on, a

    # (t1_us already inferred above when None)

    # Global edge times: union of interval boundaries within window.
    edge_times = {float(t0_us), float(t1_us)}
    for segs in intervals.values():
        for a, b, _amp in segs:
            if t0_us <= a <= t1_us:
                edge_times.add(float(a))
            if t0_us <= b <= t1_us:
                edge_times.add(float(b))
    edge_times = sorted(edge_times)

    # Decide whether to insert a "-" column between adjacent edge times.
    if gap_threshold_us is None:
        # "Extended time": 10% of window or 5 us, whichever is smaller, but at least 0.5 us.
        gap_threshold_us = max(0.5, min(5.0, 0.10 * max(0.0, float(t1_us - t0_us))))

    # Columns are edge timestamps only (no "-" spacer columns), but we enforce
    # the rule that a new column only exists if at least one lane has a rising
    # or falling edge between the previous and current timestamp.
    full_columns: List[float] = [float(t) for t in edge_times]
    columns: List[float] = []
    if full_columns:
        columns.append(full_columns[0])
        for c in full_columns[1:]:
            prev_c = columns[-1]
            changed = False
            for _label, kind, ch in rows:
                on_prev, _ = state_amp_at(kind, int(ch), float(prev_c))
                on_curr, _ = state_amp_at(kind, int(ch), float(c))
                if bool(on_prev) != bool(on_curr):
                    changed = True
                    break
            if changed:
                columns.append(c)

    # Build matrices using the filtered columns.
    state_rows = []
    amp_rows = []
    for label, kind, ch in rows:
        # Include units in row labels.
        if kind == "gen":
            unit = "DAC units" if dac_units else "norm"
            label_amp = f"{label} ({unit})"
            label_state = f"{label}"
        else:
            # ADC window is a gate (no amplitude in DAC units).
            label_amp = f"{label} (ADC gate)"
            label_state = f"{label}"

        srow = [label_state]
        arow = [label_amp]
        current_on = None
        current_amp = None
        for col in columns:
            if col == "-":
                # Use last known state at previous edge.
                on = bool(current_on) if current_on is not None else False
                amp = float(current_amp) if current_amp is not None else 0.0
            else:
                on, amp = state_amp_at(kind, int(ch), float(col))
                current_on, current_amp = on, amp
            srow.append("on" if on else "off")
            # Amplitude matrix: numbers only. Off -> 0.
            arow.append(f"{amp:.6g}" if on else "0")
        state_rows.append(srow)
        amp_rows.append(arow)

    # Write CSVs
    import csv as _csv

    state_path = f"{out_prefix}_state.csv"
    amp_path = f"{out_prefix}_amp.csv"
    # First column is the timestamp row label; other columns are timestamps (ns).
    # Use the minimum number of significant figures needed so that distinct edge
    # times get distinct labels. This avoids artificial "(2)" suffixes when two
    # nearby times round to the same value at low precision.
    ns_values = [float(c) * 1_000.0 for c in columns]
    header_cols = []
    # Try increasing numbers of decimal places until the formatted labels are unique.
    for decimals in range(2, 10):
        labels = []
        for v in ns_values:
            s = f"{v:.{decimals}e}"
            mant, exp = s.split("e")
            mant = mant.rstrip("0").rstrip(".")
            exp_i = int(exp)  # removes padding like e+02 -> 2
            labels.append(f"{mant}e{exp_i}")
        if len(labels) == len(set(labels)):
            header_cols = labels
            break
    else:
        # Fallback: if we somehow still collide, append indices to keep them unique.
        seen = {}
        for lbl in labels:
            n = seen.get(lbl, 0) + 1
            seen[lbl] = n
            header_cols.append(lbl if n == 1 else f"{lbl}({n})")
    header = ["timestamp (ns)"] + header_cols

    with open(state_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(header)
        w.writerows(state_rows)

    with open(amp_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(header)
        w.writerows(amp_rows)

    return state_path, amp_path


def csv_to_table_png(csv_path: str, png_path: str, title: str = "") -> None:
    """
    Render a CSV (e.g. edge matrix) as a PNG table with optional highlighting.

    Rows/columns are taken from the CSV. Cells with "on" or numeric value > 0
    are highlighted light blue.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    def _display_col(col: str) -> str:
        if col == df.columns[0]:
            return col
        s = str(col).strip()
        suffix = ""
        if "(" in s and s.endswith(")"):
            base, suf = s.rsplit("(", 1)
            base = base.strip()
            suf = "(" + suf
            try:
                float(base)
                s = base
                suffix = suf
            except Exception:
                pass
        return s + suffix

    display_cols = [_display_col(c) for c in df.columns]
    fig_h = max(2.5, 0.55 * (len(df) + 1))
    fig_w = max(8.0, 0.8 * len(df.columns))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=display_cols,
        loc="center",
        cellLoc="center",
    )
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
        if r == 0 or c == -1:
            continue
        try:
            val = df.iat[r - 1, c]
        except Exception:
            continue
        if isinstance(val, str):
            v = val.strip().lower()
            if v == "on":
                cell.set_facecolor(highlight)
            else:
                try:
                    if float(v) > 0:
                        cell.set_facecolor(highlight)
                except Exception:
                    pass
        else:
            try:
                if float(val) > 0:
                    cell.set_facecolor(highlight)
            except Exception:
                pass
    if title:
        ax.set_title(title, pad=6)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def show_schedule(
    prog,
    title: str = "Pulse schedule",
    show_amplitude: bool = True,
    amplitude_units: str = "dac",
    gen_ch_labels: Optional[dict] = None,
    physical_port_labels: Optional[dict] = None,
) -> None:
    """
    Quick interactive display of a pulse schedule (no file output).

    Use this for live visualization while connected to the RFSoC.

    Parameters
    ----------
    prog : QickProgramV2
        Compiled QICK asm_v2 program.
    title : str
        Plot title.
    show_amplitude : bool
        If True, include amplitude vs time panel.
    amplitude_units : str
        "dac" for DAC units, "norm" for normalized 0-1.
    gen_ch_labels : dict, optional
        Map gen_ch (int) -> label str for y-axis labels.
    physical_port_labels : dict, optional
        Map RFDC IDs -> human labels for port annotations.

    Example
    -------
    >>> from qcvt import show_schedule
    >>> prog = MyProgram(soccfg, reps=1, cfg=config)
    >>> show_schedule(prog, title="My experiment")
    """
    plot_pulse_schedule(
        prog,
        ax=None,
        gen_ch_labels=gen_ch_labels,
        physical_port_labels=physical_port_labels,
        show_readout_triggers=True,
        show_amplitude=show_amplitude,
        amplitude_units=amplitude_units,
        title=title,
    )
    plt.tight_layout()
    plt.show()


def visualize_all(
    prog,
    out_dir: str,
    title: str = "Pulse schedule",
    show_amplitude: bool = True,
    amplitude_units: str = "dac",
    t0_us: float = 0.0,
    t1_us: Optional[float] = None,
    rows: Optional[List[Tuple[str, str, int]]] = None,
    gen_ch_labels: Optional[dict] = None,
    physical_port_labels: Optional[dict] = None,
    schedule_dpi: int = 150,
    table_dpi: int = 200,
    show: bool = False,
) -> dict:
    """
    Generate all pulse visualization outputs in one call.

    This is a convenience function that produces:
      - schedule.png: pulse schedule plot (optionally with amplitude panel)
      - amplitudes.csv / amplitudes.npz: raw amplitude traces
      - edges_state.csv / edges_amp.csv: edge matrices
      - edges_state.png / edges_amp.png: rendered table images

    Parameters
    ----------
    prog : QickProgramV2
        Compiled QICK asm_v2 program.
    out_dir : str
        Output directory. Created if it doesn't exist.
    title : str
        Title for the schedule plot.
    show_amplitude : bool
        If True, include amplitude vs time panel in schedule plot.
    amplitude_units : str
        "dac" for DAC units (0 to maxv), "norm" for normalized 0-1.
    t0_us : float
        Start time for amplitude/edge exports (microseconds).
    t1_us : float, optional
        End time for exports. If None, inferred from schedule.
    rows : list of (label, kind, ch), optional
        Row specification for edge matrices. If None, auto-generated from
        all gen/adc channels in the schedule.
    gen_ch_labels : dict, optional
        Map gen_ch (int) -> label str for y-axis labels.
    physical_port_labels : dict, optional
        Map RFDC IDs -> human labels for port annotations.
    schedule_dpi : int
        DPI for schedule PNG (default 150).
    table_dpi : int
        DPI for table PNGs (default 200).
    show : bool
        If True, display the schedule plot interactively after saving.

    Returns
    -------
    dict
        Dictionary with paths to all generated files:
        {
            "schedule_png": str,
            "amplitudes_csv": str,
            "amplitudes_npz": str,
            "edges_state_csv": str,
            "edges_amp_csv": str,
            "edges_state_png": str,
            "edges_amp_png": str,
        }

    Example
    -------
    >>> from qcvt import visualize_all, load_soccfg_from_json
    >>> soccfg = load_soccfg_from_json("qick_config.json")
    >>> prog = MyProgram(soccfg, reps=1, cfg=config)
    >>> outputs = visualize_all(prog, "output/", title="Qubit spectroscopy")
    >>> print(outputs["schedule_png"])
    output/schedule.png
    """
    import os

    os.makedirs(out_dir, exist_ok=True)

    results = {}

    # 1. Schedule plot
    schedule_path = os.path.join(out_dir, "schedule.png")
    plot_pulse_schedule(
        prog,
        ax=None,
        gen_ch_labels=gen_ch_labels,
        physical_port_labels=physical_port_labels,
        show_readout_triggers=True,
        show_amplitude=show_amplitude,
        amplitude_units=amplitude_units,
        title=title,
    )
    plt.tight_layout()
    plt.savefig(schedule_path, dpi=schedule_dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close("all")
    results["schedule_png"] = schedule_path

    # 2. Amplitude traces CSV/NPZ
    amplitudes_csv_path = os.path.join(out_dir, "amplitudes.csv")
    try:
        npz_path = export_amplitude_traces_csv(
            prog,
            csv_path=amplitudes_csv_path,
            t0_us=t0_us,
            t1_us=t1_us if t1_us is not None else _infer_schedule_end_us(prog, t0_us),
            amplitude_units=amplitude_units,
        )
        results["amplitudes_csv"] = amplitudes_csv_path
        results["amplitudes_npz"] = npz_path
    except RuntimeError:
        results["amplitudes_csv"] = None
        results["amplitudes_npz"] = None

    # 3. Edge matrices CSV
    edges_prefix = os.path.join(out_dir, "edges")
    try:
        state_csv, amp_csv = export_edge_matrices_csv(
            prog,
            out_prefix=edges_prefix,
            t0_us=t0_us,
            t1_us=t1_us,
            rows=rows,
            amplitude_units=amplitude_units,
        )
        results["edges_state_csv"] = state_csv
        results["edges_amp_csv"] = amp_csv

        # 4. Edge matrices as PNG tables
        state_png = edges_prefix + "_state.png"
        amp_png = edges_prefix + "_amp.png"
        csv_to_table_png(state_csv, state_png, "State Edge Summary")
        csv_to_table_png(amp_csv, amp_png, "Amplitude Edge Summary")
        results["edges_state_png"] = state_png
        results["edges_amp_png"] = amp_png
    except RuntimeError:
        results["edges_state_csv"] = None
        results["edges_amp_csv"] = None
        results["edges_state_png"] = None
        results["edges_amp_png"] = None

    return results


def _infer_schedule_end_us(prog, t0_us: float) -> float:
    """Infer the end time (in us) from the schedule for amplitude export."""
    schedule = _extract_schedule(prog)
    if not schedule:
        return t0_us + 1.0

    soccfg = getattr(prog, "soccfg", None)

    def to_us(t_cy: float, length_cy: float, ch: int, kind: str):
        if soccfg is None or not hasattr(soccfg, "cycles2us"):
            return float(t_cy) / 1000.0, float(length_cy) / 1000.0
        if kind == "gen":
            return soccfg.cycles2us(t_cy, gen_ch=ch), soccfg.cycles2us(length_cy, gen_ch=ch)
        return soccfg.cycles2us(t_cy, ro_ch=ch), soccfg.cycles2us(length_cy, ro_ch=ch)

    ends = []
    for ch, _name, t_cy, length_cy, kind in schedule:
        t_us, length_us = to_us(t_cy, length_cy, int(ch), kind)
        ends.append(float(t_us + length_us))

    return max(ends, default=t0_us + 1.0)
