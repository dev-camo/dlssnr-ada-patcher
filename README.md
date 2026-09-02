# DLSS Neural Rendering Ada Patcher

Patch to enable DLSS Neural Rendering on an NVIDIA Ada GPU, such as an RTX 40
Series card.

This patcher takes a copy of `nvngx_dlssnr.dll` that you already have and adds
Ada support. It does not include, download, or distribute NVIDIA DLLs or model
data.

> **Important:** This changes a proprietary game file and could be detected by
> anti-cheat systems. Do not use the patched DLL with online, competitive, or
> anti-cheat-protected games. File-integrity systems can block the game, and
> anti-cheat systems can take account action, including a ban.

## What you need

- Python 3.10 or later.
- CUDA Toolkit 13.3, with `ptxas`, `fatbinary`, and `cuobjdump` available.
- An original copy of `nvngx_dlssnr.dll`

## Quick start

1. Open a terminal in this project directory.
2. Run the patcher with the path to the original DLL:

   ```text
   python dlssnr_ada_patcher.py "C:\path\to\nvngx_dlssnr.dll"
   ```

The patcher compiles and checks every change before it writes a file.
It renames the original file to `nvngx_dlssnr.dll.bak` and installs
the patched file under the original name.

## Contributing

Bug reports and pull requests are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md)
before getting started.

## License

This project is licensed under the GNU General Public License v2.0. See
[LICENSE](LICENSE) for the full terms.
