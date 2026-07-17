# QCVT — QICK Control Visualization Tool

Visualize and export the pulse schedule of a [QICK](https://github.com/openquantumhardware/qick)
`asm_v2` program **before it is sent to an RFSoC**, so you can confirm timing,
durations, amplitudes and sweeps are what you intended. Works online (connected)
and fully offline (from a saved config or a compiled-program pickle).

QCVT reads a *compiled* program (an `AveragerProgramV2` instance) and turns it
into a sweep-aware `Schedule`, expressed entirely in **microseconds**, that drives
the plots and exports. Timing is taken directly from QICK's own time parameters
(`t_params`) and pulse lengths (`get_length()`), and `Delay` instructions are
accumulated to recover absolute times — so what you see matches what the board
plays, even across generators and readouts running at different clock rates.

![Example schedule](examples/example_schedule.png)

## Install

```bash
pip install -e .            # core: matplotlib, numpy, pandas, cloudpickle
pip install -e ".[qick]"    # also install qick (needed to build programs / load soccfg)
```

## Quick start

### Live view while running experiments

```python
from qcvt import show_schedule

prog = YourProgram(soccfg, reps=1, final_delay=0, cfg=config)
show_schedule(prog, title="My experiment")   # interactive; no files written
```

### Everything at once

```python
from qcvt import visualize_all

outputs = visualize_all(prog, out_dir="output/", title="Qubit spectroscopy",
                        show_amplitude=True)
# outputs -> {schedule_png, amplitudes_csv, amplitudes_npz,
#             edges_state_csv, edges_amp_csv, edges_state_png, edges_amp_png}
```

### From a compiled-program pickle (no RFSoC, no qick needed to plot)

```python
from qcvt import visualize_from_pickle

prog, ax = visualize_from_pickle("compiled_program.pkl", output_path="schedule.png")
```

### Rebuild a program offline from a saved config

```python
from qcvt import save_soccfg_to_json, load_soccfg_from_json, show_schedule

# once, while connected:
save_soccfg_to_json(soc, "qick_config.json")

# later, offline:
soccfg = load_soccfg_from_json("qick_config.json")
prog = YourProgram(soccfg, reps=1, final_delay=0, cfg=config)
show_schedule(prog)
```

See `examples/run_offline_example.py` for a complete, runnable example (it uses the
bundled `examples/qick_config.json`).

### Command line

```bash
qcvt --pickle prog.pkl --out-dir ./out --show-amplitude
```

Writes `schedule.png`, `amplitudes.csv/.npz`, `edges_state.csv/.png` and
`edges_amp.csv/.png`.

## What the plot shows

- One lane per generator and per readout channel, on a shared microsecond axis.
- Each pulse as a labelled bar; readout integration windows as green bars.
- **Periodic** (CW) pulses hatched and extended to the next event on their channel.
- **Swept** parameters (time, length, gain) drawn as translucent ranges and tagged
  in the pulse label; an optional amplitude panel shows gain sweeps as a band.
- Correct amplitudes for `const`, `arb` (envelope) and `flat_top` pulses.

## API reference

| Function | Returns | Description |
|----------|---------|-------------|
| `show_schedule(prog, ...)` | `None` | Interactive display (no files saved) |
| `visualize_all(prog, out_dir, ...)` | `dict` | Schedule PNG + amplitude CSV/NPZ + edge matrices + table PNGs |
| `plot_pulse_schedule(prog, ...)` | `ax` or `(ax, ax_amp)` | Draw the schedule (and optional amplitude panel) |
| `visualize_from_pickle(path, ...)` | `(prog, ax)` | Load a compiled-program pickle and plot |
| `extract_schedule(prog)` | `Schedule` | Sweep-aware, microsecond schedule model |
| `export_amplitude_traces_csv(prog, csv, t0, t1, ...)` | `str` | Amplitude samples to CSV (+ `.npz`) |
| `export_edge_matrices_csv(prog, prefix, t0, t1, ...)` | `(str, str)` | State and amplitude edge matrices |
| `csv_to_table_png(csv, png, title)` | `None` | Render a CSV as a highlighted table |
| `save_soccfg_to_json(soc, path)` | `None` | Save RFSoC config for offline use |
| `load_soccfg_from_json(path)` | `QickConfig` | Load config (requires `qick`) |

The package is organized into `qcvt.model` (schedule extraction), `qcvt.plotting`,
`qcvt.export` and `qcvt.io`; `qcvt.pulse_visualizer` remains as a compatibility
shim for older imports.

## Notes and limitations

- The program must be compiled (an `AveragerProgramV2` compiles on construction).
- A single iteration of each loop is drawn; swept values are annotated and their
  ranges shown rather than unrolled.
- Extraction is best-effort: an unrecognized macro is skipped with a warning
  rather than aborting the whole schedule.

## License

MIT.
