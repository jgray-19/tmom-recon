Branch-comparison evidence
==========================

This report records measured acceptance evidence, rather than inferring branch
quality from a partial or differently configured run.  The worktrees were
created from the local branches on 2026-08-20 and left detached and clean under
``/tmp/tmom-branch-matrix``; the working checkout was not used for these runs.

Environment
-----------

* Python: 3.11.4
* MAD-NG: ``eb3748f``
* pymadng-utils: ``b881423``
* xtrack-tools: ``366fb5c``
* Xtrack: ``18d8bc168``

``accpy`` is an installed environment rather than a Git checkout, so it has no
revision to record.  These revisions are the common shell environment used by
the preflight; a full matrix must reuse them.

Collection preflight
--------------------

The identical command ``pytest --collect-only -q`` was used in each worktree.

===========  =========  ===============  ======================================
branch       commit     collection       result
===========  =========  ===============  ======================================
master       b68d36e    none             collection error
dev          9bf8d34    171 tests        passed (0.43 s pytest time)
extra-dev    a2ee8ed    180 tests        passed (0.62 s pytest time)
===========  =========  ===============  ======================================

``master`` fails before collection because it imports
``run_ac_dipole_tracking_with_particles`` from the installed ``xtrack_tools``;
that symbol is absent from the common checkout.  This is a branch/API
compatibility finding, not a physics-test failure and not evidence that
``extra-dev`` is accepted.

Acceptance status
-----------------

The full ``pytest -q`` matrix, diagnostic-ladder result, dependency revision
capture, ruff, diff check, and documentation build have not yet been run.
Accordingly, no branch-improvement or legacy-test-removal claim is made here.
