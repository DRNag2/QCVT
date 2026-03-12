"""
Basic tests for QCVT schedule extraction and export.

Run with: pytest tests/ -v
"""
from __future__ import annotations

import sys
import os

# Repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest


def test_import():
    """Package and main functions are importable."""
    import qcvt
    assert hasattr(qcvt, "plot_pulse_schedule")
    assert hasattr(qcvt, "export_edge_matrices_csv")
    assert hasattr(qcvt, "export_amplitude_traces_csv")
    assert hasattr(qcvt, "visualize_from_pickle")
    assert hasattr(qcvt, "save_soccfg_to_json")
    assert hasattr(qcvt, "load_soccfg_from_json")


def test_extract_schedule_empty():
    """_extract_schedule returns empty list for object without macro_list."""
    from qcvt.pulse_visualizer import _extract_schedule

    class Empty:
        macro_list = []
        pulses = {}
        soccfg = None

    assert _extract_schedule(Empty()) == []


def test_scalar_value():
    """_scalar_value resolves numbers and param-like objects."""
    from qcvt.pulse_visualizer import _scalar_value

    assert _scalar_value(3.0) == 3.0
    assert _scalar_value(10) == 10.0


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(os.path.dirname(__file__), "..", "examples", "qick_config.json")),
    reason="qick_config.json not in examples/",
)
def test_load_soccfg():
    """Load soccfg from examples if present."""
    from qcvt import load_soccfg_from_json

    path = os.path.join(os.path.dirname(__file__), "..", "examples", "qick_config.json")
    soccfg = load_soccfg_from_json(path)
    assert soccfg is not None
