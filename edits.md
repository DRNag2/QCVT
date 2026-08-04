# QCVT — implementation spec

> **Status: all tasks below are implemented and verified** (2026-08-03).
> This document is kept as the record of what was found and why each change
> was made. Post-implementation hardening: `representative_gain` picks the
> sweep endpoint with the largest |gain| (not `gain_max`, which is ≈0 for a
> negative-going sweep like -0.6..0); the amplitude band spans |gain| min→max
> (0 when the sweep crosses zero); and pulse labels are placed on the segment
> visible inside the window/inset so zoomed figures don't blow up on save.

Findings from an offline audit against `qick==0.2.422`, using `examples/qick_config.json`
and a reconstruction of the Power_rabi experiment from the original bug report.
All claims were re-verified against `qick==0.2.388` (the version installed in this
environment) and hold there too: `Resync` compiles to `TIME inc_ref` and exposes
`t_params["t"]`, `TimedMacro.__init__` sets `t_params`, and the only timed macros
are `Pulse`, `Trigger`, `Delay`, `Wait`, `Resync`, `ConfigReadout`.

**Verified as already fixed — do not regress:** `_body` relative timing is correct.
Qubit pulse length resolves to 1.6211 µs (intended 1.62), the readout pulse starts
0.0081 µs after the qubit pulse ends (intended 0.01), the ADC integration starts
0.4753 µs after the readout pulse starts (intended 0.474), and both readout and
integration resolve to exactly 30 µs. Residuals are tProc cycle quantization applied
by QICK itself, not QCVT error. `Delay` accumulation, `delay_auto` resolution,
`t='auto'` pulses, `Wait` being ignored, `_ro_length_us`, and the `flat_top`
three-segment split were all cross-checked against QICK's own source and are correct.

Tasks below are ordered by priority. Task 1 is the only one that silently corrupts
output today.

---

## Task 1 — Swept-gain pulses render and export at zero amplitude

**Priority: high. This breaks the single most common calibration experiment.**

### Problem

`param_nominal()` in `src/qcvt/model.py` returns `x.start` for a swept `QickParam`.
A power Rabi sweeps gain from 0 upward, so the nominal gain of the pulse under test
is `0.0`. Consequences:

- `amplitude_trace(prog, event)` returns an all-zero trace for that pulse.
- `_gen_intervals()` in `src/qcvt/export.py` has `if amp == 0.0: continue`, so the
  pulse is **silently dropped** from `amplitudes.csv`, `amplitudes.npz`,
  `edges_state.csv` and `edges_amp.csv`.

Reproduced: exporting a power Rabi with a swept gen 6 pulse and a fixed gen 4 pulse
produces a CSV containing only a `gen_4` column. No warning is emitted.

The amplitude *panel* in `plotting.py` is unaffected because it explicitly passes
`gain_override=e.gain_min` / `e.gain_max` (lines ~412–423) to draw a band. Keep that
behaviour — the fix below must not change it.

### Fix

In `src/qcvt/model.py`, add a helper near the other `PulseEvent` utilities:

```python
def representative_gain(event: "PulseEvent") -> float:
    """A single gain value suitable for drawing/exporting one pulse.

    For a swept gain the nominal value is the sweep *start*, which is 0.0 for a
    power Rabi — that would render the pulse under test as a flat zero line. Use
    the sweep maximum instead so the pulse is visible at its largest extent.
    """
    return event.gain_max if event.gain_swept else event.gain
```

In `amplitude_trace()`, change the gain resolution so an explicit override still
wins but the swept default is no longer the sweep start:

```python
# before
gain = abs(event.gain if gain_override is None else gain_override)

# after
gain = abs(representative_gain(event) if gain_override is None else gain_override)
```

No change needed in `export.py` — `_gen_intervals()` calls `amplitude_trace()`
without an override, so it picks up the new default and the `if amp == 0.0: continue`
guard stops firing for swept pulses while still dropping genuinely-zero pulses.

### Acceptance

- `amplitude_trace()` on a pulse with `gain=QickSweep1D("gainloop", 0.0, 1.0)` returns
  a trace whose peak is `gain_max * maxv`, not `0.0`.
- The exported CSV for a power Rabi contains a column for the swept generator.
- The amplitude panel still draws a min→max band for swept-gain pulses (unchanged).

---

## Task 2 — `Resync` silently misplaces every subsequent event

**Priority: high. Produces a confidently wrong plot with no warning.**

### Problem

`qick.asm_v2.Resync.expand()` emits `TIME inc_ref`, i.e. it advances the tProc
reference exactly like `Delay` does, and `Resync.preprocess()` calls
`prog.decrement_timestamps(delay_rounded)` — QICK's own bookkeeping treats it as a
delay of `t`. `extract_schedule()` has no branch for it, so the reference is never
advanced and everything after a `resync()` is drawn too early.

Reproduced: two 1 µs pulses separated by `self.resync(5.0)` both extract to
`t_start = 0.9993` — stacked on top of each other, no warning.

### Fix

In `extract_schedule()` in `src/qcvt/model.py`, extend the `Delay` branch:

```python
if cname in ("Delay", "Resync"):
```

Add a comment recording the semantics, because they are not identical:

```python
# Resync advances the reference like Delay (both compile to TIME inc_ref), but at
# runtime it applies max(0, t - elapsed), so the drawn position is an upper bound.
# QICK's own timestamp bookkeeping uses the full t, so we match it.
```

Emit a one-time warning per program so the user knows the plot is an upper bound:

```python
if cname == "Resync" and not _resync_warned:
    warnings.warn("QCVT: program contains Resync; times after it are upper bounds "
                  "(Resync applies max(0, t - elapsed) at runtime).")
    _resync_warned = True
```

### Acceptance

- Two pulses separated by `resync(5.0)` extract 5 µs apart (within cycle rounding).
- A warning is emitted once, not once per macro.

---

## Task 3 — Unrecognized macros are skipped silently, contradicting the README

**Priority: high. This is what let Task 2 go unnoticed.**

### Problem

The README claims: *"an unrecognized macro is skipped with a warning rather than
aborting the whole schedule."* This is false. `warnings.warn` only fires inside the
`except` block, so a macro type with no `if`/`elif` branch falls through in complete
silence. Any future QICK release that adds a timed macro will silently degrade the
schedule.

### Fix

Distinguish timed from untimed macros by duck-typing — `qick.asm_v2.TimedMacro.__init__`
sets `self.t_params = {}`, and the plain `Macro` base does not. Do **not** import qick.

In `src/qcvt/model.py`, add:

```python
# Timed macros that legitimately place nothing on the timeline.
_IGNORED_TIMED_MACROS = frozenset({"Wait", "ConfigReadout"})
```

At the end of the `for macro in macro_list` chain, replace the bare comment with a
real `else`:

```python
else:
    # Register ops, loop control and labels carry no timing. Anything with
    # t_params is a TimedMacro we don't know about — that's a real gap.
    if hasattr(macro, "t_params") and cname not in _IGNORED_TIMED_MACROS:
        if cname not in _unknown_warned:
            warnings.warn(
                f"QCVT: unhandled timed macro {cname!r}; schedule may be "
                f"incomplete or misaligned after this point."
            )
            _unknown_warned.add(cname)
```

Initialize `_unknown_warned = set()` and `_resync_warned = False` at the top of
`extract_schedule()` so the state is per-call, not module-global.

### Acceptance

- A program containing an unhandled `TimedMacro` subclass emits exactly one warning
  naming the class.
- `Wait`, `ConfigReadout`, `OpenLoop`, `CloseLoop`, `WriteReg`, `IncReg`, `Label`,
  `End` produce no warnings.

---

## Task 4 — `suppressed_events()` matches `"off"` as a bare substring

**Priority: low.**

### Problem

`Schedule.suppressed_events()` in `src/qcvt/model.py` does:

```python
if not e.periodic and ("turnoff" in n or "off" in n):
```

`"off" in n` matches `offset_cal`, `off_resonant_probe`, `readout_offset` and similar.
Such a pulse is hidden from the plot if it shares a start time with a periodic pulse
on the same channel. Narrow blast radius, but a silent disappearance.

### Fix

```python
import re

_OFF_PULSE_RE = re.compile(r"(^|[_\-\s])(turn[_\-\s]?)?off$")
```

(The originally proposed pattern `(^|[_\-\s])(turn)?off($|[_\-\s])` fails its own
acceptance: `off_resonant_probe` starts with the standalone token `off` and would
still be suppressed. The acceptance list implies "off" must be the *final* token,
which is what the anchored pattern above enforces.)

```python
if not e.periodic and _OFF_PULSE_RE.search(n):
```

### Acceptance

- `pump_off`, `turnoff`, `turn_off`, `off` are still suppressed.
- `offset_cal`, `off_resonant_probe` are not.

---

## Task 5 — Optional body-relative time origin

**Priority: medium. Not a bug; prevents false bug reports.**

### Problem

The schedule uses an absolute program timeline, so with the default
`initial_delay=1.0` a pulse written as `self.pulse(..., t=0)` inside `_body()` is
drawn at ~2 µs. This is correct, but it does not match how anyone reads their own
`_body()`, and it is exactly the kind of apparent discrepancy that generated the
original bug report.

### Fix

Add a field to `Schedule`:

```python
body_start_us: float = 0.0
```

In `extract_schedule()`, capture the reference at the first `OpenLoop` — everything
before it is `_initialize()`, everything after is loop body:

```python
elif cname == "OpenLoop":
    if not _body_started:
        sched.body_start_us = ref_nom
        _body_started = True
```

Add a `time_origin` parameter to `plot_pulse_schedule()` (and thread it through
`show_schedule`, `review_schedule`, `visualize_all`, and the CLI as `--time-origin`):

```python
time_origin: str = "program"   # "program" | "body"
```

When `"body"`, subtract `sched.body_start_us` from all plotted times and label the
x-axis `Time (µs, relative to body start)`. Do not mutate the `Schedule` — apply the
offset at draw time only, so exports remain on the absolute timeline unless the
caller asks otherwise.

### Acceptance

- `plot_pulse_schedule(prog, time_origin="body")` places the first `_body()` pulse at
  t ≈ 0.
- Default behaviour is unchanged.

---

## Task 6 — Make the test suite runnable without `PYTHONPATH`

**Priority: medium.**

`python -m pytest tests/` currently fails with `ModuleNotFoundError: No module named
'qcvt'` unless `PYTHONPATH=src` is set. Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

Also add `qick` to the dev extra so the offline regression tests can actually run:

```toml
dev = ["pytest>=7", "qick"]
```

---

## Task 7 — Add the offline regression harness

**Priority: high. This is what closes the "we can't verify without hardware" gap.**

The harness is already in place at `tests/test_power_rabi_timing.py`. It reconstructs
the exact Power_rabi program from the original bug report, compiles it offline against
`examples/qick_config.json` (resolved relative to the test file, so it runs from any
working directory), and asserts the three timing relationships plus durations and
periodic-pulse extension. Verified in this environment: 4 passed, 2 xfailed.

It contains two `@pytest.mark.xfail(strict=True)` tests covering Tasks 1 and 2. The
`strict` flag means that once those tasks are implemented the tests will fail as
XPASS until you **remove the `xfail` markers** — they should then pass outright.
Do not delete the tests.

Existing coverage gap worth closing while you're in there: none of the current 11 tests
exercise a swept-gain pulse through the amplitude or export path, which is why Task 1
survived a full rewrite.

---

## Task 8 — README corrections

In `README.md`:

1. **Fix the false claim.** Under "Notes and limitations", replace
   *"an unrecognized macro is skipped with a warning"* with an accurate description
   once Task 3 lands: unhandled **timed** macros warn; untimed macros (register ops,
   loop control, labels) are ignored by design.

2. **Document the sweep representative value.** State that swept-gain pulses are drawn
   and exported at their sweep maximum, and that the amplitude panel shows the full
   min→max band.

3. **Document `Resync`.** Note that times after a `Resync` are upper bounds.

4. **Add an offline verification section.** This is the most valuable addition — it
   removes the assumption that hardware is needed to test QCVT:

   ```markdown
   ## Verifying QCVT without an RFSoC

   QICK separates compilation from execution: `AveragerProgramV2` resolves all
   timing at construction, in pure software, given only a `QickConfig`. The board
   is required to *play* pulses, never to *decide when they play*.

   ```bash
   pip install -e ".[qick,dev]"
   pytest tests/ -v
   ```

   `tests/test_power_rabi_timing.py` compiles a full Power_rabi program against the
   bundled `examples/qick_config.json` and asserts pulse-to-pulse offsets to within
   tProc cycle quantization. Any timing regression is catchable on a laptop.
   ```

5. **Document `time_origin`** in the API table once Task 5 lands.

---

## Verification checklist

Run after implementing:

```bash
pip install -e ".[qick,dev]"
pytest tests/ -v          # expect all green, no xfails remaining
```

Manual spot-check on the Power_rabi reconstruction:

- swept gen 6 pulse appears in `amplitudes.csv` with a non-zero peak
- `resync(5.0)` separates two pulses by 5 µs
- an unhandled timed macro produces exactly one warning
- `time_origin="body"` puts the first `_body()` pulse at t ≈ 0