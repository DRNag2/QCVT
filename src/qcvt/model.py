# -*- coding: utf-8 -*-
"""
Data model and schedule extraction for QCVT.

The core idea: a compiled QICK ``asm_v2`` program stores everything we need to
draw a pulse schedule, but in a form that is awkward to consume directly.  This
module turns a compiled program into a small, explicit, sweep-aware
:class:`Schedule` made of :class:`PulseEvent` objects, all expressed in
**microseconds**.

Why microseconds (and not cycles)?  QICK stores each timed instruction's time as
a ``QickParam`` in microseconds (``macro.t_params[...]``) and each pulse's length
in microseconds (``pulse.get_length()``).  Working in microseconds sidesteps the
per-channel clock conversions (generators, readouts and the tProc all run at
different clock rates) that are a common source of off-by-a-clock-ratio bugs, and
lets every channel share one correct time axis.

Absolute timing: pulses are scheduled at a *local* time ``t`` relative to a moving
reference.  ``Delay`` instructions advance that reference (their stored ``t`` is
the fully resolved delay, including ``delay_auto``), so we accumulate delays as we
walk the macro list to recover absolute times.  ``Wait`` stalls the processor but
does not move the reference, so it is ignored for placement.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# QickParam helpers (work on plain numbers too, so nothing here requires qick)
# --------------------------------------------------------------------------- #
def _is_qickparam(x: Any) -> bool:
    """True if ``x`` looks like a QickParam (has ``.start`` and ``.spans``)."""
    return hasattr(x, "start") and hasattr(x, "spans")


def param_nominal(x: Any) -> float:
    """Return a single representative float from a QickParam or number."""
    if _is_qickparam(x):
        try:
            return float(x.start)
        except Exception:
            try:
                return float(x.minval())
            except Exception:
                return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def param_range(x: Any) -> Tuple[float, float, bool]:
    """Return ``(min, max, is_swept)`` for a QickParam or number (same units)."""
    if _is_qickparam(x):
        swept = bool(getattr(x, "spans", None))
        try:
            lo, hi = float(x.minval()), float(x.maxval())
        except Exception:
            v = param_nominal(x)
            lo = hi = v
        return lo, hi, swept
    v = param_nominal(x)
    return v, v, False


# Backwards-compatible alias used by older code/tests.
def _scalar_value(x: Any) -> float:
    return param_nominal(x)


def _finite(*vals: float) -> bool:
    return all(np.isfinite(v) for v in vals)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class PulseEvent:
    """A single generator pulse or ADC integration window, in microseconds."""

    ch: int
    name: str
    kind: str  # "gen" or "adc"
    t_start: float  # nominal absolute start time (us)
    length: float  # nominal length (us)
    t_min: float = 0.0
    t_max: float = 0.0
    len_min: float = 0.0
    len_max: float = 0.0
    style: str = "const"
    envelope: Optional[str] = None
    periodic: bool = False
    gain: float = 0.0
    gain_min: float = 0.0
    gain_max: float = 0.0
    freq: Optional[float] = None
    phase: Optional[float] = None
    swept_params: Tuple[str, ...] = ()

    @property
    def t_end(self) -> float:
        return self.t_start + self.length

    @property
    def time_swept(self) -> bool:
        return not np.isclose(self.t_min, self.t_max)

    @property
    def length_swept(self) -> bool:
        return not np.isclose(self.len_min, self.len_max)

    @property
    def gain_swept(self) -> bool:
        return not np.isclose(self.gain_min, self.gain_max)


@dataclass
class Schedule:
    """A normalized, sweep-aware view of a compiled QICK program."""

    events: List[PulseEvent] = field(default_factory=list)
    soccfg: Any = None
    prog: Any = None
    loop_dict: Dict[str, int] = field(default_factory=dict)

    @property
    def gen_events(self) -> List[PulseEvent]:
        return [e for e in self.events if e.kind == "gen"]

    @property
    def adc_events(self) -> List[PulseEvent]:
        return [e for e in self.events if e.kind == "adc"]

    @property
    def gen_chs(self) -> List[int]:
        return sorted({e.ch for e in self.gen_events if e.ch >= 0})

    @property
    def adc_chs(self) -> List[int]:
        return sorted({e.ch for e in self.adc_events})

    def __bool__(self) -> bool:
        return bool(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def end_us(self) -> float:
        """Nominal end time of the last event (us)."""
        return max((e.t_end for e in self.events), default=1.0)

    def draw_lengths(self, window_end_us: Optional[float] = None) -> Dict[int, float]:
        """Resolve display lengths, extending ``periodic`` pulses to the next
        event on the same channel (or the window end).  Keyed by ``id(event)``.
        """
        if window_end_us is None:
            window_end_us = self.end_us()
        out: Dict[int, float] = {}
        by_ch: Dict[int, List[PulseEvent]] = {}
        for e in self.gen_events:
            by_ch.setdefault(e.ch, []).append(e)
        for ch, evs in by_ch.items():
            evs_sorted = sorted(evs, key=lambda e: e.t_start)
            for i, e in enumerate(evs_sorted):
                if not e.periodic:
                    out[id(e)] = e.length
                    continue
                # Extend to the next strictly-later event on this channel.
                k = i + 1
                while k < len(evs_sorted) and evs_sorted[k].t_start <= e.t_start + 1e-9:
                    k += 1
                nxt = evs_sorted[k].t_start if k < len(evs_sorted) else window_end_us
                out[id(e)] = max(0.0, nxt - e.t_start)
        return out

    def suppressed_events(self) -> set:
        """Return ids of events to hide: a non-periodic "off"/"turnoff" pulse
        scheduled at the same time as a periodic pulse on the same channel
        (a common cleanup artifact that would otherwise clutter the plot).
        """
        skip = set()
        by_key: Dict[Tuple[int, float], List[PulseEvent]] = {}
        for e in self.gen_events:
            by_key.setdefault((e.ch, round(e.t_start, 9)), []).append(e)
        for evs in by_key.values():
            if len(evs) < 2:
                continue
            if any(e.periodic for e in evs):
                for e in evs:
                    n = e.name.lower()
                    if not e.periodic and ("turnoff" in n or "off" in n):
                        skip.add(id(e))
        return skip


# --------------------------------------------------------------------------- #
# Pulse parameter lookups
# --------------------------------------------------------------------------- #
def _macro_time_param(macro: Any, name: str):
    """Return the (rounded, sweep-aware) time QickParam for ``name`` in us."""
    getter = getattr(macro, "get_time_param", None)
    if callable(getter):
        try:
            return getter(name)
        except Exception:
            pass
    return getattr(macro, "t_params", {}).get(name)


def _pulse_param_range(prog: Any, name: str, param: str) -> Tuple[float, float, float, bool]:
    """Return ``(nominal, min, max, is_swept)`` for a pulse parameter.

    Prefers ``prog.get_pulse_param`` (fully rounded, loop-aware); falls back to
    the raw ``pulse.params`` entry.
    """
    getter = getattr(prog, "get_pulse_param", None)
    if callable(getter):
        try:
            arr = np.asarray(getter(name, param, as_array=True), dtype=float).ravel()
            if arr.size:
                lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
                return float(arr.flat[0]), lo, hi, not np.isclose(lo, hi)
        except Exception:
            pass
    try:
        p = getattr(prog, "pulses", {})[name].params.get(param)
    except Exception:
        p = None
    if p is None:
        return 0.0, 0.0, 0.0, False
    lo, hi, swept = param_range(p)
    return param_nominal(p), lo, hi, swept


def _ro_length_us(prog: Any, ro: int) -> Optional[float]:
    """ADC integration-window length (us) for readout channel ``ro``."""
    try:
        rc = prog.ro_chs[ro]
        length = rc["length"]
        f_output = prog.soccfg["readouts"][ro]["f_output"]
        return float(length) / float(f_output)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract_schedule(prog: Any) -> Schedule:
    """Build a :class:`Schedule` from a compiled QICK ``asm_v2`` program.

    The program must be compiled (``AveragerProgramV2`` compiles on construction).
    Timing is recovered in microseconds and is sweep-aware; on any unexpected
    structure the offending event is skipped rather than aborting the whole
    schedule.
    """
    sched = Schedule(soccfg=getattr(prog, "soccfg", None), prog=prog,
                     loop_dict=dict(getattr(prog, "loop_dict", {}) or {}))

    macro_list = getattr(prog, "macro_list", None) or []
    pulses = getattr(prog, "pulses", None) or {}
    if not macro_list:
        return sched

    # Moving reference offset (us), tracked with its sweep range.
    ref_nom = ref_min = ref_max = 0.0

    for macro in macro_list:
        cname = type(macro).__name__
        try:
            if cname == "Delay":
                tp = _macro_time_param(macro, "t")
                if tp is None:
                    continue
                lo, hi, _ = param_range(tp)
                ref_nom += param_nominal(tp)
                ref_min += lo
                ref_max += hi

            elif cname == "Pulse":
                ch = getattr(macro, "ch", None)
                name = getattr(macro, "name", None)
                if ch is None or name is None or name not in pulses:
                    continue
                tp = _macro_time_param(macro, "t")
                if tp is None:
                    continue
                t_nom = param_nominal(tp)
                t_lo, t_hi, _ = param_range(tp)
                pulse = pulses[name]
                length_qp = pulse.get_length()
                l_nom = param_nominal(length_qp)
                l_lo, l_hi, _ = param_range(length_qp)
                if not _finite(t_nom, l_nom) or l_nom < 0:
                    continue
                params = getattr(pulse, "params", {}) or {}
                style = str(params.get("style", "const"))
                envelope = params.get("envelope")
                periodic = params.get("mode") == "periodic"
                g_nom, g_lo, g_hi, g_sw = _pulse_param_range(prog, name, "gain")
                f_nom, _, _, f_sw = _pulse_param_range(prog, name, "freq")
                p_nom, _, _, _ = _pulse_param_range(prog, name, "phase")
                swept = []
                if not np.isclose(t_lo, t_hi):
                    swept.append("time")
                if not np.isclose(l_lo, l_hi):
                    swept.append("length")
                if g_sw:
                    swept.append("gain")
                if f_sw:
                    swept.append("freq")
                sched.events.append(PulseEvent(
                    ch=int(ch), name=str(name), kind="gen",
                    t_start=ref_nom + t_nom, length=l_nom,
                    t_min=ref_min + t_lo, t_max=ref_max + t_hi,
                    len_min=l_lo, len_max=l_hi,
                    style=style, envelope=envelope, periodic=periodic,
                    gain=g_nom, gain_min=g_lo, gain_max=g_hi,
                    freq=f_nom, phase=p_nom, swept_params=tuple(swept),
                ))

            elif cname == "Trigger":
                ros = getattr(macro, "ros", None) or []
                if not ros:
                    continue
                tp = _macro_time_param(macro, "t")
                if tp is None:
                    continue
                t_nom = param_nominal(tp)
                t_lo, t_hi, _ = param_range(tp)
                width_qp = _macro_time_param(macro, "width")
                for ro in ros:
                    # Prefer the true integration length; fall back to trigger width.
                    length = _ro_length_us(prog, int(ro))
                    if length is None:
                        length = param_nominal(width_qp) if width_qp is not None else 0.0
                    if not _finite(t_nom, length) or length < 0:
                        continue
                    sched.events.append(PulseEvent(
                        ch=int(ro), name="readout", kind="adc",
                        t_start=ref_nom + t_nom, length=float(length),
                        t_min=ref_min + t_lo, t_max=ref_max + t_hi,
                        len_min=float(length), len_max=float(length),
                        style="const",
                    ))
            # Wait / ConfigReadout / loops / register ops: no timeline placement.
        except Exception as exc:  # keep going; a single bad macro shouldn't kill the plot
            warnings.warn(f"QCVT: skipping macro {cname}: {exc}")

    return sched


# --------------------------------------------------------------------------- #
# Amplitude reconstruction
# --------------------------------------------------------------------------- #
def _gencfg(prog: Any, ch: int) -> dict:
    try:
        return dict(prog.soccfg["gens"][ch])
    except Exception:
        return {}


def amplitude_trace(prog: Any, event: PulseEvent, length_us: Optional[float] = None,
                    dac_units: bool = True, gain_override: Optional[float] = None):
    """Return ``(t_us, amp)`` samples describing one pulse's amplitude envelope.

    * ``const``    -> a rectangle at ``|gain| * scale`` for the pulse length.
    * ``flat_top`` -> a rectangle (the flat plateau dominates; edges are short
      ramps that are not resolved here).
    * ``arb``      -> the stored envelope magnitude, scaled by ``|gain| * scale``.

    ``scale`` is ``maxv`` (DAC units) when ``dac_units`` else 1.0 (normalized).
    ``length_us`` overrides the pulse length (used for periodic extension).
    """
    if length_us is None:
        length_us = event.length
    t0 = event.t_start
    gain = abs(event.gain if gain_override is None else gain_override)
    gencfg = _gencfg(prog, event.ch)
    maxv = int(gencfg.get("maxv", 32766))
    scale = maxv if dac_units else 1.0

    def _box():
        return (np.array([t0, t0, t0 + length_us, t0 + length_us]),
                np.array([0.0, gain * scale, gain * scale, 0.0]))

    if event.style in ("const", "flat_top") or not event.envelope:
        return _box()

    envelopes = getattr(prog, "envelopes", None)
    try:
        data = np.asarray(envelopes[event.ch]["envs"][event.envelope]["data"])
    except Exception:
        return _box()
    if data.ndim == 2 and data.shape[1] >= 2:
        mag = np.hypot(data[:, 0].astype(float), data[:, 1].astype(float))
    else:
        mag = np.abs(data.astype(float))
    if mag.size == 0:
        return _box()

    # Envelope samples are at the generator's DAC sample rate fs (MHz), so the
    # per-sample spacing is 1/fs.  fs == f_fabric * samps_per_clk.
    fs = float(gencfg.get("fs", 0.0))
    if fs <= 0:
        f_fabric = float(gencfg.get("f_fabric", 1000.0)) or 1000.0
        samps_per_clk = float(gencfg.get("samps_per_clk", 1)) or 1.0
        fs = f_fabric * samps_per_clk
    dt_us = 1.0 / fs
    t = t0 + np.arange(mag.size) * dt_us
    peak = float(np.max(mag)) or 1.0
    amp = (mag / peak) * gain * scale
    # Anchor the trace to the baseline at both ends for clean fills/plots.
    t = np.concatenate([[t0], t, [t[-1]]])
    amp = np.concatenate([[0.0], amp, [0.0]])
    return t, amp
