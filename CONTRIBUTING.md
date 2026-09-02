# Contributing

Thanks for helping improve DLSS Neural Rendering Ada Patcher. Bug reports,
documentation fixes, tests, and focused code changes are welcome.

## Before you start

- Search the existing issues before opening a new one.
- Use the appropriate issue form for bugs, feature proposals, or questions.
- Open an issue before investing in a large change or a change to supported
  binary or PTX formats. This helps confirm that the approach fits the
  project's scope.
- Report potential vulnerabilities according to [SECURITY.md](SECURITY.md),
  not in a public issue.

Small, self-contained fixes do not need an issue first.

## Proprietary files and safe use

Do not commit, upload, or link to NVIDIA DLLs, model data, patched DLLs,
extracted PTX, cubins, fatbins, or other game files. Tests must use synthetic
fixtures. Logs and screenshots should have personal paths and unrelated data
redacted.

Patching invalidates the DLL's NVIDIA Authenticode signature. Do not use a
patched DLL with online, competitive, or anti-cheat-protected software. This
project will not help bypass anti-cheat or file-integrity systems.

## Development setup

The project supports Python 3.10 and later and uses only the Python standard
library. There is no dependency installation or build step.

From the repository root, run the complete unit suite:

```text
python -m unittest discover -s tests -v
```

Smoke-test the command-line interface with:

```text
python dlssnr_ada_patcher.py --help
```

Run a focused test while iterating, for example:

```text
python -m unittest tests.test_patcher.PtxTests
python -m unittest tests.test_patcher.PtxTests.test_transform_supported_operations
```

The tests do not require CUDA or a proprietary DLL. An optional end-to-end run
requires CUDA Toolkit 13.3 and a DLL that you obtained yourself. Start with
`--dry-run`, and never include the input or generated files in an issue or pull
request.

## Code expectations

Match the existing annotated Python style and retain Python 3.10 compatibility.
Prefer standard-library solutions and focused changes over new dependencies or
broad refactors.

This project modifies binary data, so unfamiliar input must fail closed:

- Validate offsets, lengths, layouts, and expected match counts before changing
  a shared `bytearray`.
- Raise `PatchError` with an actionable message for expected input, filesystem,
  and external-tool failures.
- Keep PTX rewrites narrow. Every newly accepted instruction shape needs both a
  success test and a rejection test for nearby unsupported input.
- Preserve source ELF and PTX images when repacking fatbins. The generated Ada
  cubin must remain additive, and rebuilt containers must fit their original
  allocations.
- Route CUDA subprocesses through `run_tool`.
- Preserve the staged atomic-write, backup, collision-check, and rollback
  behavior.

Use synthetic byte fixtures, `tempfile.TemporaryDirectory`, and
`unittest.mock` in tests. A bug fix should include a regression test whenever
practical.

Update `README.md` when a change affects prerequisites, command-line syntax,
output behavior, or safety guidance.

## Pull requests

Keep each pull request focused on one problem. In its description:

1. Explain the problem and the chosen solution.
2. Link any related issue.
3. List the exact validation commands you ran and their results.
4. Call out checks you could not run, compatibility assumptions, and any
   end-to-end testing performed without sharing proprietary artifacts.

Complete the pull request template, respond to review feedback, and keep new or
changed behavior covered by tests. Maintainers may ask to narrow or revise a
change when its safety cannot be established from the available fixtures.

## Licensing

By submitting a contribution, you agree that it may be distributed under the
GNU General Public License v2.0 included in [LICENSE](LICENSE).
