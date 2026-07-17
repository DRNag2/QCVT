"""Tests for QCVT schedule extraction, plotting and exports.

Run with: pytest tests/ -v
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

# Make the in-tree package importable without an editable install.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

CONFIG = os.path.join(os.path.dirname(__file__), "..", "examples", "qick_config.json")
try:
    import qick  # noqa: F401
    HAVE_QICK = True
except Exception:
    HAVE_QICK = False

needs_qick = pytest.mark.skipif(
    not (HAVE_QICK and os.path.isfile(CONFIG)),
    reason="requires qick and examples/qick_config.json",
)


def test_public_api():
    import qcvt

    for name in [
        "plot_pulse_schedule", "show_schedule", "visualize_all",
        "visualize_from_pickle", "extract_schedule", "Schedule", "PulseEvent",
        "export_edge_matrices_csv", "export_amplitude_traces_csv",
        "csv_to_table_png", "save_soccfg_to_json", "load_soccfg_from_json",
        "review_schedule",
    ]:
        assert hasattr(qcvt, name), name


def test_extract_schedule_empty():
    from qcvt import extract_schedule

    class Empty:
        macro_list = []
        pulses = {}
        soccfg = None

    sched = extract_schedule(Empty())
    assert len(sched) == 0
    assert not sched


def test_param_helpers():
    from qcvt.model import param_nominal, param_range

    assert param_nominal(3.0) == 3.0
    assert param_range(5)[:2] == (5.0, 5.0)
    assert param_range(5)[2] is False

    class FakeSweep:
        start = 1.0
        spans = {"loop": 4.0}

        def minval(self):
            return 1.0

        def maxval(self):
            return 5.0

    lo, hi, swept = param_range(FakeSweep())
    assert (lo, hi, swept) == (1.0, 5.0, True)


def test_legacy_extract_tuple_shim():
    from qcvt.pulse_visualizer import _extract_schedule

    class Empty:
        macro_list = []
        pulses = {}
        soccfg = None

    assert _extract_schedule(Empty()) == []


# --------------------------------------------------------------------------- #
# Golden tests against a real (offline-built) program
# --------------------------------------------------------------------------- #
def _build_spec_program():
    from qick.asm_v2 import AveragerProgramV2, QickSweep1D
    from qcvt import load_soccfg_from_json

    soccfg = load_soccfg_from_json(CONFIG)

    class Spec(AveragerProgramV2):
        def _initialize(self, cfg):
            self.declare_gen(ch=2, nqz=2)
            self.declare_gen(ch=6, nqz=2)
            self.add_loop("freqloop", 11)
            self.declare_readout(ch=0, length=10.0)
            self.add_readoutconfig(ch=0, name="ro", freq=1000, gen_ch=6)
            self.add_pulse(ch=2, name="qpulse", ro_ch=0, style="const",
                           length=5.0, freq=QickSweep1D("freqloop", 3000, 3200),
                           phase=0, gain=0.3)
            self.add_pulse(ch=6, name="readout", ro_ch=0, style="const",
                           length=10.0, freq=1000, phase=0, gain=0.5)

        def _body(self, cfg):
            self.send_readoutconfig(ch=0, name="ro", t=0)
            self.pulse(ch=2, name="qpulse", t=0)
            self.delay_auto(0.01)
            self.pulse(ch=6, name="readout", t=0)
            self.trigger(ros=[0], pins=[0], t=0.5)

    return Spec(soccfg, reps=2, final_delay=100, cfg={}, reps_innermost=False)


@needs_qick
def test_timing_and_sweeps():
    from qcvt import extract_schedule

    sched = extract_schedule(_build_spec_program())
    by_name = {e.name: e for e in sched.gen_events}

    q = by_name["qpulse"]
    assert q.length == pytest.approx(5.0, abs=1e-2)
    assert q.t_start == pytest.approx(1.0, abs=1e-2)  # includes initial sync delay
    assert "freq" in q.swept_params

    r = by_name["readout"]
    # readout follows qpulse + delay_auto(0.01): ~1.0 + 5.0 + 0.01
    assert r.t_start == pytest.approx(6.01, abs=5e-2)
    assert r.length == pytest.approx(10.0, abs=1e-2)

    adc = sched.adc_events[0]
    assert adc.t_start == pytest.approx(6.51, abs=5e-2)
    # ADC window uses the readout integration length, not the tiny trigger width.
    assert adc.length == pytest.approx(10.0, abs=1e-1)


@needs_qick
def test_plot_with_ax_and_amplitude_no_crash():
    import matplotlib.pyplot as plt
    from qcvt import plot_pulse_schedule

    prog = _build_spec_program()
    fig, ax = plt.subplots()
    result = plot_pulse_schedule(prog, ax=ax, show_amplitude=True)
    assert isinstance(result, tuple) and len(result) == 2
    plt.close("all")


@needs_qick
def test_visualize_all_writes_all_outputs(tmp_path):
    from qcvt import visualize_all

    prog = _build_spec_program()
    out = visualize_all(prog, str(tmp_path), title="spec", show_amplitude=True)
    for key in ("schedule_png", "amplitudes_csv", "amplitudes_npz",
                "edges_state_csv", "edges_amp_csv", "edges_state_png", "edges_amp_png"):
        assert out[key] and os.path.isfile(out[key]), key


@needs_qick
def test_edge_matrix_amplitude_values(tmp_path):
    import csv

    from qcvt import export_edge_matrices_csv

    prog = _build_spec_program()
    state_csv, amp_csv = export_edge_matrices_csv(
        prog, out_prefix=str(tmp_path / "edges"), t0_us=0.0, t1_us=None,
    )
    with open(amp_csv) as f:
        rows = list(csv.reader(f))
    labels = [r[0] for r in rows[1:]]
    assert any("gen 2" in lbl for lbl in labels)
    # readout const gain 0.5 -> 0.5 * maxv should appear somewhere in its row.
    readout_row = next(r for r in rows[1:] if r[0].startswith("gen 6"))
    vals = [float(x) for x in readout_row[1:] if x not in ("", "0")]
    assert vals and max(vals) == pytest.approx(0.5 * 32766, rel=0.02)


def _build_flat_top_program():
    from qick.asm_v2 import AveragerProgramV2
    from qcvt import load_soccfg_from_json

    soccfg = load_soccfg_from_json(CONFIG)

    class FlatTopProg(AveragerProgramV2):
        def _initialize(self, cfg):
            self.declare_gen(ch=2, nqz=2)
            self.add_gauss(ch=2, name="ramp", sigma=0.05, length=0.3, even_length=True)
            self.add_pulse(ch=2, name="ft", style="flat_top", envelope="ramp",
                           freq=3000, phase=0, gain=0.5, length=2.0)

        def _body(self, cfg):
            self.pulse(ch=2, name="ft", t=0.5)

    return FlatTopProg(soccfg, reps=1, final_delay=1, cfg={}, reps_innermost=False)


@needs_qick
def test_flat_top_amplitude_has_ramps_and_plateau():
    from qcvt import extract_schedule
    from qcvt.model import amplitude_trace

    prog = _build_flat_top_program()
    sched = extract_schedule(prog)
    e = next(x for x in sched.gen_events if x.name == "ft")
    assert e.style == "flat_top"
    assert e.length == pytest.approx(2.3, abs=0.05)  # plateau 2.0 + ramps ~0.3

    t, amp = amplitude_trace(prog, e, dac_units=True)
    assert t is not None and amp is not None
    # Should have rising samples, a high plateau, then falling samples.
    peak = float(amp.max())
    assert peak == pytest.approx(0.5 * 32766, rel=0.05)
    # Plateau: many consecutive samples near the peak.
    on_plateau = amp > 0.95 * peak
    assert on_plateau.sum() >= 2
    # Ramps: amplitude takes intermediate values, not just 0 and peak.
    mid = (amp > 0.05 * peak) & (amp < 0.95 * peak)
    assert mid.sum() >= 4
    # Total span matches get_length.
    assert (t[-1] - t[0]) == pytest.approx(e.length, abs=0.05)


@needs_qick
def test_review_schedule_saves_and_returns_true(tmp_path):
    from qcvt import review_schedule

    prog = _build_spec_program()
    ok = review_schedule(
        prog,
        save_dir=str(tmp_path / "review"),
        title="review test",
        show=False,
        confirm=False,
    )
    assert ok is True
    assert os.path.isfile(tmp_path / "review" / "schedule.png")


@needs_qick
def test_multi_timescale_window_and_insets():
    """Short ns-scale pulse next to a long readout still plots; zoom window works."""
    import matplotlib.pyplot as plt
    from qick.asm_v2 import AveragerProgramV2
    from qcvt import load_soccfg_from_json, plot_pulse_schedule, extract_schedule

    soccfg = load_soccfg_from_json(CONFIG)

    class Mixed(AveragerProgramV2):
        def _initialize(self, cfg):
            self.declare_gen(ch=2, nqz=2)
            self.declare_gen(ch=6, nqz=2)
            self.declare_readout(ch=0, length=50.0)
            self.add_readoutconfig(ch=0, name="ro", freq=1000, gen_ch=6)
            self.add_gauss(ch=2, name="g", sigma=0.01, length=0.05)
            self.add_pulse(ch=2, name="short", style="arb", envelope="g",
                           freq=3200, phase=0, gain=0.8)
            self.add_pulse(ch=6, name="long", style="const", length=50.0,
                           freq=1000, phase=0, gain=0.4)

        def _body(self, cfg):
            self.pulse(ch=2, name="short", t=0.1)
            self.delay_auto(0.05)
            self.pulse(ch=6, name="long", t=0)
            self.trigger(ros=[0], pins=[0], t=0.2)

    prog = Mixed(soccfg, reps=1, final_delay=10, cfg={}, reps_innermost=False)
    sched = extract_schedule(prog)
    short = next(e for e in sched.gen_events if e.name == "short")
    long = next(e for e in sched.gen_events if e.name == "long")
    assert short.length < 0.2
    assert long.length == pytest.approx(50.0, abs=0.1)

    # Full window + forced inset must not crash.
    ax, ax_amp = plot_pulse_schedule(prog, show_amplitude=True, insets=True)
    assert ax.get_xlim()[1] > 40
    plt.close("all")

    # Zoomed window around the short pulse.
    ax = plot_pulse_schedule(prog, show_amplitude=False, t0_us=0.0, max_time_us=1.0, insets=False)
    assert ax.get_xlim() == pytest.approx((0.0, 1.0))
    plt.close("all")
