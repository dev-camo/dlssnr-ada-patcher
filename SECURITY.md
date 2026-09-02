# Security Policy

## Supported versions

The project does not currently publish versioned releases. Security fixes are
made on the `main` branch, so only the latest revision is supported.

## Reporting a vulnerability

Do not include sensitive details in a public issue. Email vulnerability reports
to [security@camo.dev](mailto:security@camo.dev).

Include, when available:

- A concise description of the vulnerability and its impact.
- The affected revision and environment.
- Reproduction steps using synthetic or freely redistributable data.
- Any suggested mitigation or fix.

Never send or attach an NVIDIA DLL, model data, extracted PTX, cubin, fatbin,
patched DLL, or other proprietary game file. Redact personal paths and unrelated
system information from logs.

The maintainer will validate the report and coordinate a fix and disclosure as
availability permits. Please allow time for a response before publishing an
unfixed vulnerability.

## Scope

Security reports should concern the patcher itself, including untrusted-input
parsing, command execution, path handling, and output installation. Anti-cheat
or file-integrity detection is an expected consequence of modifying a signed
DLL, not a vulnerability in this project. Do not test against systems or files
that you do not own or have permission to use.
