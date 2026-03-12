"""
CLI for QCVT: plot and export from a compiled program pickle.
"""
from __future__ import annotations

import argparse
import os
import sys


def csv_to_table_png(csv_path: str, png_path: str, title: str = "") -> None:
    """
    Render a CSV (e.g. edge matrix) as a PNG table with optional highlighting.

    Rows/columns are taken from the CSV. Cells with "on" or numeric value > 0
    are highlighted light blue.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="QCVT: plot pulse schedule and export edge matrices from a compiled QICK program pickle.",
    )
    parser.add_argument(
        "--pickle",
        required=True,
        help="Path to compiled program pickle (.pkl)",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Output directory for schedule PNG and edge CSVs/PNGs (default: current dir)",
    )
    parser.add_argument(
        "--title",
        default="Pulse schedule",
        help="Title for the schedule plot",
    )
    parser.add_argument(
        "--show-amplitude",
        action="store_true",
        help="Add amplitude vs time panel to the schedule plot",
    )
    parser.add_argument(
        "--no-table-png",
        action="store_true",
        help="Do not render edge matrices as PNG tables (only write CSVs)",
    )
    parser.add_argument(
        "--t0",
        type=float,
        default=0.0,
        help="Start time for export window (µs)",
    )
    parser.add_argument(
        "--t1",
        type=float,
        default=None,
        help="End time for export window (µs); default: infer from schedule",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.pickle):
        print("Error: pickle file not found:", args.pickle, file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)

    try:
        import cloudpickle
    except ImportError:
        print("Error: cloudpickle is required. Install with: pip install cloudpickle", file=sys.stderr)
        return 1

    with open(args.pickle, "rb") as f:
        prog = cloudpickle.load(f)

    from qcvt import (
        plot_pulse_schedule,
        export_amplitude_traces_csv,
        export_edge_matrices_csv,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot schedule
    fig_out = os.path.join(args.out_dir, "schedule.png")
    plot_pulse_schedule(
        prog,
        show_amplitude=args.show_amplitude,
        amplitude_units="dac",
        title=args.title,
    )
    plt.tight_layout()
    plt.savefig(fig_out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print("Saved", fig_out)

    # Edge matrices (CSV)
    prefix = os.path.join(args.out_dir, "edges")
    state_csv, amp_csv = export_edge_matrices_csv(
        prog,
        out_prefix=prefix,
        t0_us=args.t0,
        t1_us=args.t1,
        rows=None,
    )
    print("Saved", state_csv)
    print("Saved", amp_csv)

    if not args.no_table_png:
        state_png = prefix + "_state.png"
        amp_png = prefix + "_amp.png"
        csv_to_table_png(state_csv, state_png, "State Edge Summary")
        csv_to_table_png(amp_csv, amp_png, "Amplitude Edge Summary")
        print("Saved", state_png)
        print("Saved", amp_png)

    return 0


if __name__ == "__main__":
    sys.exit(main())
