# pylhc GitHub Organization — Changes Since January 1, 2026

Covers releases and merged PRs from **2026-01-01 through 2026-08-11** for the active repos in https://github.com/pylhc.
---

## omc3

- **0.29.0** (Jul 23, 2026) — Harpy parallelisation across bunches: new `n_jobs` parameter with auto-determined worker count, BLAS threads capped at 1, RAM-aware fallback to a single worker. US→UK spelling conversion pass.
- **0.28.1** (Mar 27, 2026) — Dropped Python 3.10 support (min now 3.11). Fixed systematic-error loading bug in the Analytical N-BPM method (sextupole horizontal misalignment was doubled instead of squared). Added beta/alpha function documentation.
- **0.28.0** (Mar 5, 2026) — KMOD measurement summary generation with logbook posting. New SuperKEKB BPM re-synchronisation script. Added `orbit` parameter for fake-measurement generation. Machine-settings extraction from NXCALS and LSA.
- **0.27.0** (Jan 31, 2026) — New MAD-NG response-matrix module for faster derivative calculations. Fake-measurement script now also outputs orbit files.
- **0.26.1** (Jan 31, 2026) — Pandas 3.x compatibility (copy-on-write fixes). Migrated CRDT fitting from deprecated `scipy.odr` to `odrpack`.

Merged PRs (notable, includes those folded into the releases above):
- **#588** "More Ruff" — fsoubelet (Aug 10, 2026) — **unreleased**
- **#587** "Update CHANGELOG.md to release today" — jgray-19 (Jul 23, 2026) — release-prep for 0.29.0
- **#585** "[Harpy] Parallelisation Strategy" — fsoubelet (Jul 22, 2026) — core of 0.29.0
- **#582** "Converting US spelling to UK spelling" — jgray-19 (Jun 24, 2026)
- **#581** "Deprecation and warning fixes" — fsoubelet (Jun 19, 2026)
- **#580** "Action calculation / rescaling fix" — fsoubelet (Jun 15, 2026)
- **#578** "Patch release and dropping Python 3.10" — fsoubelet (Mar 27, 2026) — release-prep for 0.28.1
- **#577** "Fix Systematic Errors Loading in Analytical N-BPM" — fsoubelet (Mar 27, 2026) — folded into 0.28.1

Only **#588** is unreleased as of this report.

---

## turn_by_turn

Releases:
- **1.5.0** (Jul 23, 2026) — Fixed xtrack test configuration; negated horizontal BPM reading for PSB (bug fix).
- **1.4.1** (Apr 15, 2026) — Added PSB alias for GUI compatibility.
- **1.4.0** (Apr 10, 2026) — New PSB BPM-format reader.
- **1.3.1** (Mar 9, 2026) — Patch: removed a default `xtrack` import at package import time.
- **1.3.0** (Mar 6, 2026) — Added loading of tracking data from `xtrack.Line` with new `MultiElementMonitor` support; removed the old `xtrack_line` module.
- **1.2.0** (Feb 12, 2026) — Added SuperKEKB turn-by-turn data support (LER and HER rings).

Merged PRs:
- **#40** "Fix xtrack tests config" — fsoubelet (Jul 22, 2026) — folded into 1.5.0
- **#41** "Negate horizontal BPM reading for PSB" — jgray-19 (Jul 22, 2026) — folded into 1.5.0
- **#39** "Add alias to psb, for GUI compatibility" — jgray-19 (Apr 15, 2026) — folded into 1.4.1
- **#38** "PSB Reader" — jgray-19 (Apr 10, 2026) — folded into 1.4.0
- **#37** "Fix default xtrack import" — fsoubelet (Mar 9, 2026) — folded into 1.3.1
- **#35** "Update pyproject.toml" — jgray-19 (Mar 6, 2026)
- **#34** "xtrack docs improvement" — jgray-19 (Mar 6, 2026)
- **#33** "Add compatibility for new xtrack feature" — jgray-19 (Mar 6, 2026) — folded into 1.3.0
- **#32** "Add a SuperKEKB reader" — Mael-Le-Garrec (Feb 12, 2026) — folded into 1.2.0

---

## pylhc.github.io (org website — no formal releases)

- **Run logbook entries** (LHC/PS/PSB 2026 run updates, screenshots, MD writeups): #280, #279, #275, #274, #273, #271, #270, #269, #264, #258, #257, #256, #255, #252, #250, #249, #248, #247, #246, #244, #243, #242, #241, #240, #239, #235
- **New features/pages**: #278 (OMC QR codes), #265 (Optics panel for BBEAT GUI), #238 (Segment-by-Segment GUI info page), #236 (LS3 planning)
- **Site infrastructure**: #237 (Zensical site-generator migration), #234 (VM setup)
- **Docs/content cleanup**: #260 (British-English conversion, matches the omc3 spelling-conversion effort), #275 (remove contacts)

---

## optics_functions

Releases:
- **0.1.6** (Jan 29, 2026) — Pandas 3.x compatibility fixes.

Merged PRs:
- **#40** "Pandas 3.x compatibility" — fsoubelet (Jan 29, 2026) — this is the 0.1.6 release itself; no further unreleased PRs found.

---

## PyLHC

Releases:
- **v0.8.4** (Mar 12, 2026) — Restricted optional `pyjapc` dependency to Python < 3.12 (upstream support ended); warns users (affects BRST logging).
- **v0.8.3** (Feb 2, 2026) — Explicit `pytz` dependency (no longer a hard pandas 3.x dependency).

Merged PRs:
- **#113** "Pyjapc pin and warning" — fsoubelet (Mar 12, 2026) — the v0.8.4 release
- **#111** "Pytz dependency specification" — fsoubelet (Feb 2, 2026) — the v0.8.3 release

---

## tfs (tfs-pandas)

Releases:
- **4.0.2** (Jan 29, 2026) — Pandas 3.x compatibility: enforces string dtype coercion for column names.
- **4.0.1** (Jan 13, 2026) — Fixed HDF5 write crash with string paths; improved type hints; added MAD-NG/pymadng feature guidance.
- **4.0.0** (Jan 3, 2026) — Major release: MAD-NG feature support, dropped Python 3.9 (min now 3.10), DataFrame validation now OFF by default for read/write, support for boolean/complex/nullable types.

No further unreleased PRs found since 4.0.2.

---

## chroma_gui

**Release:** **0.0.26** (Apr 10, 2026) — "First github release." No further PR activity was found associated with this release in the available PR listing (only PR #1 "GitHub transfer" from JoschD, dated 2025, predates the window).

### Overall summary for the window (Jan 1 – Aug 11, 2026)

- **omc3** and **turn_by_turn** are the most active library repos: omc3 shipped four releases (0.26.1 → 0.29.0) spanning pandas-3.x compatibility, a new MAD-NG response-matrix module, KMOD summaries, NXCALS/LSA extraction, an N-BPM bug fix, and Harpy parallelisation; turn_by_turn shipped six releases (1.2.0 → 1.5.0) covering SuperKEKB support, xtrack.Line tracking-data loading, a new PSB BPM reader, and a PSB horizontal-BPM sign fix. Both have one small unreleased PR each (changelog/credits, and a ruff-lint pass).
- **pylhc.github.io** (org website) had by far the most PR volume — mostly routine 2026-run logbook entries — but also notable feature work: a British-English documentation conversion effort, a new "Optics panel for BBEAT GUI", a Segment-by-Segment GUI info page, and a migration to the Zensical site generator.
- **optics_functions** and **PyLHC** each shipped a small compatibility release (pandas 3.x / pyjapc-pin+pytz respectively).
- **tfs** has two merged-but-unreleased pandas-3.x/MAD-NG PRs sitting since Jan 2026 — no new tag yet.
- **chroma_gui** got its first tagged GitHub release (0.0.26) in the window.
