# QCVT — QICK pulse schedule visualizer

Offline visualization and edge-matrix export for [QICK](https://github.com/openquantumhardware/qick-controller) `asm_v2` pulse programs. No RFSoC connection required.

## Install

From the repo root:

```bash
pip install -e .
```

With optional QICK support (needed for `load_soccfg_from_json` and building programs from config):

```bash
pip install -e ".[qick]"
```

## Quick start

### From a compiled program pickle (e.g. from a measurement)

```python
from qcvt import visualize_from_pickle

prog, ax = visualize_from_pickle("path/to/compiled_program.pkl", output_path="schedule.png")
```

### From a program built with soccfg from file

1. **Once, when connected to the RFSoC**, save the config:

   ```python
   from qcvt import save_soccfg_to_json
   save_soccfg_to_json(soc, "qick_config.json")
   ```

2. **Offline**, load config, build your program, then visualize:

   ```python
   from qcvt import load_soccfg_from_json, plot_pulse_schedule
   import matplotlib.pyplot as plt

   soccfg = load_soccfg_from_json("qick_config.json")
   prog = YourProgram(soccfg, reps=1, cfg=config)  # your QICK program class
   plot_pulse_schedule(prog, show_amplitude=True)
   plt.savefig("schedule.png")
   plt.show()
   ```

### Command-line (pickle → plot + edge matrices)

```bash
qcvt --pickle path/to/prog.pkl --out-dir ./out
```

This writes:

- `out/schedule.png` — pulse schedule (and amplitude panel if requested)
- `out/edges_state.csv`, `out/edges_amp.csv` — edge matrices
- `out/edges_state.png`, `out/edges_amp.png` — table renderings of the matrices

## API overview

| Function | Description |
|----------|-------------|
| `plot_pulse_schedule(prog, ...)` | Plot channel schedule and optional amplitude vs time |
| `export_amplitude_traces_csv(prog, csv_path, t0_us, t1_us, ...)` | Export raw amplitude samples to CSV + NPZ |
| `export_edge_matrices_csv(prog, out_prefix, t0_us, t1_us=None, rows=None, ...)` | Export state and amplitude edge matrices (CSV) |
| `save_soccfg_to_json(soc, path)` | Save RFSoC config for offline use |
| `load_soccfg_from_json(path)` | Load config (requires `qick`) |
| `visualize_from_pickle(pickle_path, output_path=None, title=None)` | Load pickle and plot |

See docstrings in `qcvt.pulse_visualizer` for full arguments.

## License

MIT.
