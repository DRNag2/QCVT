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
from qcvt import show_schedule, review_schedule

prog = YourProgram(soccfg, reps=1, final_delay=0, cfg=config)
show_schedule(prog, title="My experiment")   # interactive; no files written

# Pre-submit gate: save a PNG, optionally prompt before acquire()
ok = review_schedule(prog, save_dir="qcvt_reviews/my_exp", show=True, confirm=True)
if not ok:
    raise RuntimeError("aborted")
```

### Everything at once

```python
from qcvt import visualize_all

outputs = visualize_all(prog, out_dir="output/", title="Qubit spectroscopy",
                        show_amplitude=True)
# outputs -> {schedule_png, edges_state_csv, edges_state_png}
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

Writes `schedule.png` and `edges_state.csv/.png`.

## What the plot shows

- One lane per generator and per readout channel, on a shared microsecond axis.
- Each pulse as a labelled bar; readout integration windows as green bars.
- **Periodic** (CW) pulses hatched and extended to the next event on their channel.
- **Swept** parameters (time, length, gain) drawn as translucent ranges and tagged
  in the pulse label; an optional amplitude panel shows gain sweeps as a band.
- **Swept-gain pulses** are drawn at the sweep endpoint with the
  **largest magnitude** (QICK gains are signed, so sweeps like -0.6..0.6 work),
  making the pulse visible at its largest extent; the amplitude panel shows the
  full |gain| min→max band, which reaches 0 when a sweep crosses zero.
- **Time origin**: by default the axis is the absolute program timeline (which
  includes any initial delay from `_initialize()`).  Pass `time_origin="body"`
  (CLI: `--time-origin body`) to `plot_pulse_schedule`, `show_schedule`,
  `review_schedule` or `visualize_all` to place t = 0 at the start of the loop
  body — matching how times read inside your `_body()`.  This affects plots
  only; the state edge-matrix export always stays on the absolute timeline.
- Correct amplitudes for all QICK pulse styles:
  - ``const`` — rectangle
  - ``arb`` — curved envelopes (gaussian, DRAG, arbitrary I/Q) sampled at the DAC rate
  - ``flat_top`` — rising ramp + plateau + falling ramp (QICK's three-segment convention)
- **Multi-timescale** programs (ns qubit pulses next to µs readout / ms CW):
  - set ``t0_us`` / ``max_time_us`` to zoom the viewing window
  - short pulses that would be invisible get a tick + duration callout
  - when length dynamic range is large, an automatic zoom inset focuses on the short pulses

## API reference

| Function | Returns | Description |
|----------|---------|-------------|
| `show_schedule(prog, ...)` | `None` | Interactive display (no files saved) |
| `review_schedule(prog, save_dir=..., ...)` | `bool` | Pre-submit gate: save PNG, optional confirm/abort before acquire |
| `visualize_all(prog, out_dir, ...)` | `dict` | Schedule PNG + state edge matrix + table PNG |
| `plot_pulse_schedule(prog, ...)` | `ax` or `(ax, ax_amp)` | Draw the schedule (and optional amplitude panel) |
| `visualize_from_pickle(path, ...)` | `(prog, ax)` | Load a compiled-program pickle and plot |
| `extract_schedule(prog)` | `Schedule` | Sweep-aware, microsecond schedule model |
| `export_edge_matrix_csv(prog, prefix, t0, t1, ...)` | `str` | On/off state edge matrix CSV |
| `csv_to_table_png(csv, png, title)` | `None` | Render a CSV as a highlighted table |
| `save_soccfg_to_json(soc, path)` | `None` | Save RFSoC config for offline use |
| `load_soccfg_from_json(path)` | `QickConfig` | Load config (requires `qick`) |

The package is organized into `qcvt.model` (schedule extraction), `qcvt.plotting`,
`qcvt.export` and `qcvt.io`.

## Notes and limitations

- The program must be compiled (an `AveragerProgramV2` compiles on construction).
- A single iteration of each loop is drawn; swept values are annotated and their
  ranges shown rather than unrolled.
- Extraction is best-effort: a macro that fails to parse is skipped with a
  warning rather than aborting the whole schedule.  Unhandled **timed** macros
  (anything with `t_params` that QCVT does not recognize) emit a warning, since
  they may shift or omit events; untimed macros (register ops, loop control,
  labels) are ignored by design.
- `resync()` advances the time reference by *at most* its argument (at runtime
  it applies `max(0, t - elapsed)`), so times drawn after a `Resync` are upper
  bounds.  A warning is emitted when a program contains one.

## Verifying QCVT without an RFSoC

QICK separates compilation from execution: `AveragerProgramV2` resolves all
timing at construction, in pure software, given only a `QickConfig`. The board
is required to *play* pulses, never to *decide when they play*.

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

`tests/test_power_rabi_timing.py` compiles a full Power_rabi program against the
bundled `examples/qick_config.json` and asserts pulse-to-pulse offsets to within
tProc cycle quantization. Any timing regression is catchable on a laptop.

## License

MIT.
