# Repository Guidelines

## Project Overview

- This is a Python 3.10+ CLI patcher that adds Ada (`sm_89`) CUDA images and
  enables Ada architecture paths in a user-supplied `nvngx_dlssnr.dll`.
- `dlssnr_ada_patcher.py` contains the CLI and all PE parsing, CUDA fatbin
  handling, PTX transformation, host patching, verification, and output logic.
- `tests/test_patcher.py` contains `unittest` coverage built from synthetic
  PE/fatbin bytes, temporary directories, and mocks. `tests/__init__.py` marks
  the test package.
- The implementation and tests use only the Python standard library. There is
  no package manifest, build step, CI workflow, formatter, linter, or
  type-checker configuration in the repository.

## Validation Commands

Run commands from the repository root.

- Run the full unit suite with `python -m unittest discover -s tests -v`.
- Run one test class with
  `python -m unittest tests.test_patcher.PtxTests`.
- Run one test with
  `python -m unittest tests.test_patcher.PtxTests.test_transform_supported_operations`.
- Smoke-test argument parsing with `python dlssnr_ada_patcher.py --help`.
- Run targeted tests first while iterating, then run the full suite before
  finishing a code change. Do not claim lint, format, type-check, or CI results;
  none of those checks is configured.

The unit suite does not need a proprietary DLL or an installed CUDA toolkit. A
real patcher run requires CUDA Toolkit 13.3 with `ptxas`, `fatbinary`, and
`cuobjdump` on `PATH` or supplied through `--cuda-bin`.

## Architecture and Safety Invariants

- Keep malformed or unfamiliar inputs fail-closed with `PatchError`; binary
  parsers and PTX rewrites deliberately validate exact counts, layouts, and
  supported instruction forms before changing bytes.
- Preserve the pipeline in `patch_file`: validate the PE and host patches,
  rebuild and round-trip-check every fatbin, ensure each rebuilt container fits
  its original allocation, update the PE checksum, and only then write output.
- Preserve source ELF images and source PTX content during fatbin repacking,
  allowing only the intentional line-ending normalization. The generated Ada
  cubin is additive, and unused fatbin space is padded so later PE offsets do
  not move.
- Keep PTX transformations narrow. Add success and rejection tests for every
  newly supported instruction shape; do not silently accept an unknown form.
- `canonical_ptx` intentionally removes only carriage returns immediately
  before line feeds. This tolerates Windows text-mode expansion without hiding
  standalone carriage returns or real payload changes.
- Route CUDA subprocesses through `run_tool` so launch failures, exit codes, and
  bounded diagnostics continue to become actionable `PatchError` messages.
- Preserve staged atomic writes, input identity rechecks, collision checks,
  file modes, and rollback behavior. By default the CLI renames the input to
  `<name>.bak` and replaces it; `--output` keeps the input, and `--dry-run`
  compiles and verifies without installing output.

## Code and Test Conventions

- Follow the existing annotated Python style: use
  `from __future__ import annotations`, four-space indentation, `snake_case`
  functions and variables, `PascalCase` classes, uppercase constants,
  `pathlib.Path`, and dataclasses for structured records/results.
- Use explicit little-endian `struct` formats and bounds checks for PE and
  fatbin data. Validate before mutating a shared `bytearray`.
- Keep expected input, filesystem, and external-tool failures user-facing
  through `PatchError`; retain `main`'s nonzero exit behavior for errors and
  interrupts.
- Name tests `test_*` and group them by subsystem. Prefer synthetic byte
  fixtures, `tempfile.TemporaryDirectory`, and `unittest.mock` over checked-in
  binaries or calls to a real CUDA installation.
- Cover both valid transformations and malformed/unsupported inputs. For output
  changes, cover backup, explicit-output, collision, interruption, and
  restoration paths as applicable.

## Documentation Boundaries

- Keep `README.md` brief, general, and easy to scan for players and
  non-specialist users. Focus it on what the tool does, prerequisites,
  basic use, output behavior, and safety warnings.
- Keep coding-agent and contributor workflow instructions in `AGENTS.md`, not
  in the README.
- Put implementation details and non-obvious technical rationale in focused
  comments beside the relevant code.
- Update the README when user-facing prerequisites, CLI syntax, output behavior,
  or safety guidance changes; do not turn it into a design document.

## Proprietary and Generated Files

- Never commit or distribute NVIDIA DLLs or model data. Use only user-supplied
  copies for explicitly requested end-to-end checks.
- Do not commit patched DLLs, `.dll.bak` backups, temporary files, CUDA
  extraction/repack directories (`fatbin_*`), Python bytecode, or tool caches;
  these are excluded by `.gitignore`.
- Do not treat generated PTX, cubins, or fatbins from `--work-dir` as source
  files.
- Retain the README and CLI warnings that patching invalidates the NVIDIA
  Authenticode signature and can trigger anti-cheat or file-integrity systems.
  Do not use a patched DLL with anti-cheat-protected software.
