# DLSS Neural Rendering RTX Patcher

Patch a user-supplied `nvngx_dlssnr.dll` for NVIDIA GeForce RTX 20, 30, 40,
and 50 Series GPUs.

The script ensures the selected CUDA images are present, enables the
corresponding NGX architecture paths, verifies the rebuilt containers, and
updates the PE checksum. It does not include, download, or distribute NVIDIA
DLLs or model files.

## Requirements

- Python 3.10 or newer
- [CUDA Toolkit 13.3](https://developer.nvidia.com/cuda-downloads)
- Your own copy of `nvngx_dlssnr.dll`

`ptxas`, `fatbinary`, and `cuobjdump` must be on `PATH`. Alternatively, pass
their directory with `--cuda-bin`.

## Use

1. Copy the original DLL somewhere safe.
2. Open a terminal in this repository.
3. Run:

   ```text
   python dlssnr_ada_patcher.py "C:\path\to\nvngx_dlssnr.dll"
   ```

With no architecture flags, the patcher includes every supported generation:

| Generation | GeForce family | CUDA target | Flag |
| --- | --- | --- | --- |
| Turing | RTX 20 Series | `sm_75` | `-t`, `--turing` |
| Ampere | RTX 30 Series | `sm_86` | `-A`, `--ampere` |
| Ada | RTX 40 Series | `sm_89` | `-a`, `--ada` |
| Blackwell | RTX 50 Series | `sm_120` | `-b`, `--blackwell` |

Use one or more flags to build only their union. For example:

```text
python dlssnr_ada_patcher.py --turing --ampere "C:\path\to\nvngx_dlssnr.dll"
```

By default, the original becomes `nvngx_dlssnr.dll.bak` and the patched file
takes its place. Use `--output patched.dll` to keep the input unchanged, or
`--dry-run` to compile and verify without writing an output file.

Run `python dlssnr_ada_patcher.py --help` for every option.

## Important

Turing and Ampere do not support the source kernels' native FP8 instructions.
Their generated paths approximate those operations with FP16 instructions, so
image quality, performance, and stability can differ by GPU and application.

Patching removes the NVIDIA Authenticode certificate because any byte change
invalidates its signature. Windows will show the patched DLL as unsigned.
This can trigger anti-cheat or file-integrity systems. Do not use a patched DLL
with online, competitive, or anti-cheat-protected software.

The patcher validates the expected PE, fatbin, and PTX structures and stops on
an unfamiliar DLL or unsupported instruction form. A successful build does not
guarantee that every game or future DLL version will work. Keep the backup so
you can restore the original.

## Contributing

Bug reports and focused pull requests are welcome. Read
[CONTRIBUTING.md](CONTRIBUTING.md) before getting started.

## License

Licensed under the [GNU General Public License v2.0](LICENSE).
