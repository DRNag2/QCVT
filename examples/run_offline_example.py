"""
Example: build a minimal QICK program from a saved soccfg and visualize it.

Prerequisites:
  1. Once, when connected to the RFSoC, run:
       from qcvt import save_soccfg_to_json
       save_soccfg_to_json(soc, 'qick_config.json')
  2. Copy qick_config.json into this directory (or set CONFIG_PATH below).

Then run from the repo root:
  python examples/run_offline_example.py

Or with a compiled program pickle only (no qick needed for plotting):
  python -c "
  from qcvt import visualize_from_pickle
  visualize_from_pickle('path/to/prog.pkl', output_path='schedule.png')
  "
"""
from __future__ import annotations

import os
import sys

# Allow running from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "qick_config.json")
if not os.path.isfile(CONFIG_PATH):
    CONFIG_PATH = os.path.join(REPO_ROOT, "qick_config.json")


def main() -> int:
    try:
        from qcvt import load_soccfg_from_json, plot_pulse_schedule, export_edge_matrices_csv
    except ImportError as e:
        print("Install qcvt and optional dependency qick: pip install -e '.[qick]'", file=sys.stderr)
        raise e

    if not os.path.isfile(CONFIG_PATH):
        print("No qick_config.json found. Save it once when connected:")
        print("  from qcvt import save_soccfg_to_json")
        print("  save_soccfg_to_json(soc, 'qick_config.json')")
        return 1

    soccfg = load_soccfg_from_json(CONFIG_PATH)

    # Minimal program: one generator, one readout, one pulse
    from qick.asm_v2 import AveragerProgramV2

    class MinimalProgram(AveragerProgramV2):
        def _initialize(self, cfg):
            self.declare_gen(ch=cfg["gen_ch"], nqz=cfg["nqz"])
            self.declare_readout(ch=cfg["ro_ch"], length=cfg["ro_len"])
            self.add_readoutconfig(ch=cfg["ro_ch"], name="ro", freq=cfg["freq"], gen_ch=cfg["gen_ch"])
            self.add_pulse(
                ch=cfg["gen_ch"],
                name="p",
                ro_ch=cfg["ro_ch"],
                style="const",
                length=cfg["pulse_len"],
                freq=cfg["freq"],
                phase=0,
                gain=0.5,
            )

        def _body(self, cfg):
            self.send_readoutconfig(ch=cfg["ro_ch"], name="ro", t=0)
            self.pulse(ch=cfg["gen_ch"], name="p", t=0)
            self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["pulse_len"], ddr4=False)

    cfg = {
        "gen_ch": 0,
        "ro_ch": 0,
        "nqz": 1,
        "freq": 6000,
        "ro_len": 100,
        "pulse_len": 200,
    }
    prog = MinimalProgram(soccfg, reps=1, final_delay=0, cfg=cfg, reps_innermost=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_pulse_schedule(prog, show_amplitude=True, title="Minimal program (example)")
    out = os.path.join(os.path.dirname(__file__), "example_schedule.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved", out)

    prefix = os.path.join(os.path.dirname(__file__), "example_edges")
    export_edge_matrices_csv(prog, out_prefix=prefix, t0_us=0.0, t1_us=None)
    print("Saved", prefix + "_state.csv", "and", prefix + "_amp.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
