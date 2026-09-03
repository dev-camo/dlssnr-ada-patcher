## Summary

<!-- What problem does this PR solve, and how? Keep the change focused. -->

## Related issue

<!-- Use "Closes #123" when applicable. Write "None" if no issue is needed. -->

## Validation

<!-- List every command you ran and its result. Explain any checks not run. -->

```text
python -m unittest discover -s tests -v
python dlssnr_patcher.py --help
```

## Checklist

- [ ] I read and followed `CONTRIBUTING.md`.
- [ ] I added or updated synthetic tests for changed behavior where practical,
      including rejection cases for newly accepted binary or PTX forms.
- [ ] I preserved Python 3.10 compatibility and kept the change focused.
- [ ] I updated user documentation for changes to prerequisites, CLI syntax,
      output behavior, or safety guidance.
- [ ] This PR contains no NVIDIA/game DLLs, model data, patched binaries,
      extracted PTX, cubins, fatbins, or other proprietary/generated files.
- [ ] This change does not remove or weaken the anti-cheat and file-integrity
      warnings.
- [ ] I agree that my contribution may be distributed under the repository's
      GNU General Public License v2.0.
