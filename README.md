# DLSS Neural Rendering Patcher

DLSS Neural Rendering Patcher is a command-line tool that prepares your copy
of `nvngx_dlssnr.dll` for NVIDIA GeForce RTX 20, 30, and 40 Series GPUs.

The patcher checks its work before writing the patched file. It does not
include, download, or distribute NVIDIA DLLs or model files.

> **Safety warning:** Patching changes the DLL and invalidates its NVIDIA
> digital signature. Anti-cheat or file-integrity systems may flag the modified
> file. Do not use it with online, competitive, or anti-cheat-protected
> software.

## Requirements

- [Python 3.10 or newer](https://www.python.org/downloads/)
- [CUDA Toolkit 13.3](https://developer.nvidia.com/cuda-downloads)
  - **Windows:** If the CUDA installer reports that Visual Studio is missing,
    you can ignore the message and continue the installation. Visual Studio is
    not required to run this patcher.
  - The CUDA tools `ptxas`, `fatbinary`, and `cuobjdump` must be available on
    your system's `PATH`. If the patcher cannot find them, use `--cuda-bin` to
    point it to the CUDA Toolkit's `bin` folder.
- Your own original copy of `nvngx_dlssnr.dll`

## Basic use

1. Download this repository as a ZIP file and extract it.
2. Find your original `nvngx_dlssnr.dll` and save a separate copy somewhere
   safe.
3. Open Command Prompt, PowerShell, or a terminal in the patcher's folder.
4. Run:

   ```text
   python dlssnr_patcher.py "C:\path\to\nvngx_dlssnr.dll"
   ```

Replace the example path with the full path to your DLL. Keep the quotation
marks, especially if the path contains spaces.

With no GPU options, the patcher includes every supported generation. If you
are unsure which option to use, run the basic command above.

## Choosing GPU generations

To select specific GPU generations instead of all four, add one or more of
these options:

| GeForce family | GPU generation | Option |
| --- | --- | --- |
| RTX 20 Series | Turing | `-t`, `--turing` |
| RTX 30 Series | Ampere | `-A`, `--ampere` |
| RTX 40 Series | Ada | `-a`, `--ada` |
| RTX 50 Series | Blackwell | `-b`, `--blackwell` |

Options can be combined. For example, this command selects Turing and Ampere:

```text
python dlssnr_patcher.py --turing --ampere "C:\path\to\nvngx_dlssnr.dll"
```

## Output and backups

By default, the patcher renames the original file to
`nvngx_dlssnr.dll.bak`, then saves the patched file as `nvngx_dlssnr.dll`. It
will not overwrite an existing backup.

Use `--output patched.dll` to keep the input file unchanged. Use `--dry-run`
to check the DLL and compile the changes without writing an output file.

Run `python dlssnr_patcher.py --help` to see every option.

## Contributing

Bug reports and focused pull requests are welcome. Read
[CONTRIBUTING.md](CONTRIBUTING.md) before getting started.

## License

Licensed under the [GNU General Public License v2.0](LICENSE).
