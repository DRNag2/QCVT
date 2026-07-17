# -*- coding: utf-8 -*-
"""
Input/output helpers and one-call orchestration for QCVT.

These functions cover the two offline entry points:

* load a compiled-program pickle (as saved into a QCoDeS dataset) and visualize it;
* save/load a ``soccfg`` JSON so programs can be rebuilt and visualized without a
  live RFSoC connection.

:func:`visualize_all` extracts the schedule once and produces every artifact
(schedule PNG, amplitude CSV/NPZ, edge matrices and their table PNGs).
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

from .export import (
    csv_to_table_png,
    export_amplitude_traces_csv,
    export_edge_matrices_csv,
)
from .model import extract_schedule
from .plotting import plot_pulse_schedule

try:  # optional: only needed to build QickConfig from JSON
    from qick.qick_asm import QickConfig
except Exception:  # pragma: no cover - qick not installed
    QickConfig = None


def save_soccfg_to_json(soc, path: str) -> None:
    """Save the current RFSoC config to JSON so programs can be built offline.

    Call this once while connected::

        save_soccfg_to_json(soc, "qick_config.json")
    """
    with open(path, "w") as f:
        json.dump(soc.get_cfg(), f, indent=2)
    print(f"Saved soccfg to {path}")


def load_soccfg_from_json(path: str):
    """Load a :class:`QickConfig` from a JSON file (requires ``qick``)."""
    if QickConfig is None:
        raise ImportError("qick is required for load_soccfg_from_json")
    return QickConfig(path)


def load_program_pickle(pickle_path: str):
    """Load a compiled program from a cloudpickle/pickle file."""
    try:
        import cloudpickle as pickle_mod
    except ImportError:
        import pickle as pickle_mod
    with open(pickle_path, "rb") as f:
        return pickle_mod.load(f)


def visualize_from_pickle(
    pickle_path: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    show_amplitude: bool = True,
    amplitude_units: str = "dac",
    show: bool = False,
):
    """Load a compiled-program pickle and plot its pulse schedule.

    Parameters
    ----------
    pickle_path : str
        Path to the ``.pkl`` file (e.g. ``compiled_program_pickle`` from a dataset).
    output_path : str, optional
        If set, save the figure here.
    title : str, optional
        Plot title.
    show_amplitude : bool
        Include the amplitude panel.
    amplitude_units : str
        ``"dac"`` or ``"norm"``.
    show : bool
        Call ``plt.show()`` (defaults to ``False`` so the function is headless-safe).

    Returns
    -------
    (prog, ax)
    """
    import matplotlib.pyplot as plt

    prog = load_program_pickle(pickle_path)
    result = plot_pulse_schedule(
        prog,
        show_amplitude=show_amplitude,
        amplitude_units=amplitude_units,
        title=title or "Pulse schedule",
    )
    ax = result[0] if isinstance(result, tuple) else result
    if output_path:
        ax.figure.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    if show:
        plt.show()
    return prog, ax


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
    table_dpi: int = 200,  # accepted for backwards compatibility
    show: bool = False,
) -> dict:
    """Generate every visualization artifact for ``prog`` in ``out_dir``.

    Returns a dict of output paths (values are ``None`` when a step is skipped,
    e.g. no generator pulses for the amplitude export).
    """
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    sched = extract_schedule(prog)
    results: dict = {}

    schedule_path = os.path.join(out_dir, "schedule.png")
    result = plot_pulse_schedule(
        prog,
        schedule=sched,
        gen_ch_labels=gen_ch_labels,
        physical_port_labels=physical_port_labels,
        show_amplitude=show_amplitude,
        amplitude_units=amplitude_units,
        title=title,
    )
    ax = result[0] if isinstance(result, tuple) else result
    ax.figure.savefig(schedule_path, dpi=schedule_dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(ax.figure)
    results["schedule_png"] = schedule_path

    amplitudes_csv = os.path.join(out_dir, "amplitudes.csv")
    try:
        export_amplitude_traces_csv(
            prog, amplitudes_csv, t0_us=t0_us, t1_us=t1_us,
            amplitude_units=amplitude_units, schedule=sched,
        )
        results["amplitudes_csv"] = amplitudes_csv
        results["amplitudes_npz"] = amplitudes_csv.rsplit(".", 1)[0] + ".npz"
    except RuntimeError:
        results["amplitudes_csv"] = None
        results["amplitudes_npz"] = None

    edges_prefix = os.path.join(out_dir, "edges")
    try:
        state_csv, amp_csv = export_edge_matrices_csv(
            prog, out_prefix=edges_prefix, t0_us=t0_us, t1_us=t1_us,
            rows=rows, amplitude_units=amplitude_units, schedule=sched,
        )
        results["edges_state_csv"] = state_csv
        results["edges_amp_csv"] = amp_csv
        state_png = edges_prefix + "_state.png"
        amp_png = edges_prefix + "_amp.png"
        csv_to_table_png(state_csv, state_png, "State edge summary")
        csv_to_table_png(amp_csv, amp_png, "Amplitude edge summary")
        results["edges_state_png"] = state_png
        results["edges_amp_png"] = amp_png
    except RuntimeError:
        for k in ("edges_state_csv", "edges_amp_csv", "edges_state_png", "edges_amp_png"):
            results[k] = None

    return results
