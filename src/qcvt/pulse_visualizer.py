# -*- coding: utf-8 -*-
"""
Backwards-compatibility shim.

QCVT's implementation now lives in focused submodules (:mod:`qcvt.model`,
:mod:`qcvt.plotting`, :mod:`qcvt.export`, :mod:`qcvt.io`).  This module re-exports
the public API and a couple of legacy private helpers so that existing imports
like ``from qcvt.pulse_visualizer import plot_pulse_schedule`` keep working.
"""

from __future__ import annotations

from .model import (
    PulseEvent,
    Schedule,
    amplitude_trace,
    extract_schedule,
    param_nominal,
    param_range,
    _scalar_value,
)
from .plotting import plot_pulse_schedule, show_schedule
from .export import (
    csv_to_table_png,
    export_amplitude_traces_csv,
    export_edge_matrices_csv,
)
from .io import (
    load_program_pickle,
    load_soccfg_from_json,
    review_schedule,
    save_soccfg_to_json,
    visualize_all,
    visualize_from_pickle,
)


def _extract_schedule(prog):
    """Legacy tuple-based extractor kept for backwards compatibility.

    Returns a list of ``(ch, name, t_start_us, length_us, kind)`` tuples.
    Prefer :func:`qcvt.extract_schedule`, which returns a rich
    :class:`~qcvt.model.Schedule`.
    """
    sched = extract_schedule(prog)
    return [(e.ch, e.name, e.t_start, e.length, e.kind) for e in sched.events]


__all__ = [
    "PulseEvent",
    "Schedule",
    "extract_schedule",
    "amplitude_trace",
    "param_nominal",
    "param_range",
    "plot_pulse_schedule",
    "show_schedule",
    "export_amplitude_traces_csv",
    "export_edge_matrices_csv",
    "csv_to_table_png",
    "save_soccfg_to_json",
    "load_soccfg_from_json",
    "load_program_pickle",
    "visualize_from_pickle",
    "visualize_all",
    "review_schedule",
]
