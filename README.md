# QCVT — QICK pulse schedule visualizer

Visualization and edge-matrix export for [QICK](https://github.com/openquantumhardware/qick) `asm_v2` pulse programs. Works both online (connected to RFSoC) and offline.

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

### Online (connected to RFSoC)

Quick interactive display while running experiments:

```python
from qcvt import show_schedule

prog = YourProgram(soccfg, reps=1, cfg=config)
show_schedule(prog, title="My experiment")  # displays interactively, returns None
```

Or with `run_and_save_rfsoc_prog` in kerrcat.py:

```python
iq_list, prog = run_and_save_rfsoc_prog(Qubit_spectroscopy, config, visualize=True)
```

### Offline (no RFSoC connection)

#### From a compiled program pickle

```python
from qcvt import visualize_from_pickle

prog, ax = visualize_from_pickle("path/to/compiled_program.pkl", output_path="schedule.png")
```

#### From a program built with saved soccfg

1. **Once, when connected to the RFSoC**, save the config:

   ```python
   from qcvt import save_soccfg_to_json
   save_soccfg_to_json(soc, "qick_config.json")
   ```

2. **Offline**, load config, build your program, then visualize:

   ```python
   from qcvt import load_soccfg_from_json, show_schedule

   soccfg = load_soccfg_from_json("qick_config.json")
   prog = YourProgram(soccfg, reps=1, cfg=config)
   show_schedule(prog)  # quick interactive view
   ```

### Generate all outputs at once

Use `visualize_all()` to generate schedule plot, amplitude CSV, edge matrices, and table PNGs in one call:

```python
from qcvt import visualize_all, load_soccfg_from_json

soccfg = load_soccfg_from_json("qick_config.json")
prog = YourProgram(soccfg, reps=1, cfg=config)

outputs = visualize_all(
    prog,
    out_dir="output/",
    title="Qubit spectroscopy",
    show_amplitude=True,
    show=True,  # also display interactively
)

# outputs dict contains paths to all generated files:
# - schedule_png, amplitudes_csv, amplitudes_npz
# - edges_state_csv, edges_amp_csv
# - edges_state_png, edges_amp_png
```

### Command-line

```bash
qcvt --pickle path/to/prog.pkl --out-dir ./out
```

This writes:

- `out/schedule.png` — pulse schedule (and amplitude panel if `--show-amplitude`)
- `out/edges_state.csv`, `out/edges_amp.csv` — edge matrices
- `out/edges_state.png`, `out/edges_amp.png` — table renderings of the matrices

## API reference

| Function | Returns | Description |
|----------|---------|-------------|
| `show_schedule(prog, ...)` | `None` | Quick interactive display (no files saved) |
| `visualize_all(prog, out_dir, ...)` | `dict` | Generate all outputs (schedule, CSV, edge matrices, PNGs) |
| `plot_pulse_schedule(prog, ...)` | `ax` | Plot channel schedule and optional amplitude panel |
| `export_amplitude_traces_csv(prog, csv_path, t0_us, t1_us, ...)` | `str` | Export amplitude samples to CSV + NPZ |
| `export_edge_matrices_csv(prog, out_prefix, t0_us, t1_us, ...)` | `(str, str)` | Export state and amplitude edge matrices |
| `csv_to_table_png(csv_path, png_path, title)` | `None` | Render CSV as PNG table with highlighting |
| `save_soccfg_to_json(soc, path)` | `None` | Save RFSoC config for offline use |
| `load_soccfg_from_json(path)` | `QickConfig` | Load config (requires `qick`) |
| `visualize_from_pickle(pickle_path, ...)` | `(prog, ax)` | Load pickle and plot |

See docstrings in `qcvt.pulse_visualizer` for full arguments.

## License

MIT.
