#!/usr/bin/env python3
"""Add selected RTX CUDA images to a user-supplied DLSS Neural Rendering DLL."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

FATBIN_MAGIC = 0xBA55ED50
FATBIN_MAGIC_BYTES = struct.pack("<I", FATBIN_MAGIC)
PE_MACHINE_AMD64 = 0x8664
PE32_PLUS_MAGIC = 0x20B
MAX_COMPATIBLE_PTX_VERSION = (9, 3)


@dataclass(frozen=True)
class Architecture:
    name: str
    cuda: int
    ngx: int
    ngx_name: str

    @property
    def cuda_name(self) -> str:
        return f"sm_{self.cuda}"


TURING = Architecture("turing", 75, 0x160, "NVSDK_NGX_GPU_Arch_Turing")
AMPERE = Architecture("ampere", 86, 0x170, "NVSDK_NGX_GPU_Arch_Ampere")
ADA = Architecture("ada", 89, 0x190, "NVSDK_NGX_GPU_Arch_Ada")
BLACKWELL = Architecture(
    "blackwell", 120, 0x1B0, "NVSDK_NGX_GPU_Arch_Blackwell2"
)
SUPPORTED_ARCHITECTURES = (TURING, AMPERE, ADA, BLACKWELL)

# Keep the original Ada constants available for callers of the former helpers.
TARGET_ARCH = ADA.cuda
MAX_ADA_PTX_VERSION = MAX_COMPATIBLE_PTX_VERSION
TARGET_NGX_ARCH = ADA.ngx
TARGET_NGX_ARCH_NAME = ADA.ngx_name


class PatchError(RuntimeError):
    """Report an input or tool error that the user can correct."""


@dataclass(frozen=True)
class PeSection:
    virtual_address: int
    virtual_size: int
    raw_offset: int
    raw_size: int


@dataclass(frozen=True)
class AuthenticodeRemoval:
    offset: int
    size: int
    certificate_count: int
    trailing_data_size: int


@dataclass(frozen=True)
class FatbinLocation:
    offset: int
    header_size: int
    data_size: int

    @property
    def size(self) -> int:
        return self.header_size + self.data_size


@dataclass(frozen=True)
class FatbinRecord:
    offset: int
    kind: int
    header_size: int
    payload_size: int
    flags: int


@dataclass
class TransformStats:
    values: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, count: int) -> None:
        if count:
            self.values[name] = self.values.get(name, 0) + count

    def summary(self) -> str:
        if not self.values:
            return "direct target conversion"
        return ", ".join(f"{name}={count}" for name, count in self.values.items())


@dataclass(frozen=True)
class CudaTools:
    cuobjdump: Path
    ptxas: Path
    fatbinary: Path


@dataclass(frozen=True)
class Image:
    kind: str
    architecture: int
    path: Path


@dataclass
class ArchitectureBuild:
    architecture: Architecture
    cubin_size: int
    generated: bool
    stats: TransformStats = field(default_factory=TransformStats)


@dataclass(frozen=True)
class ArchitectureRequirementPatch:
    interface: str
    offset: int
    previous_value: int


@dataclass(frozen=True)
class ArchitectureCasePatch:
    offset: int
    success_offset: int
    architecture: str = ADA.name


@dataclass(frozen=True)
class HostPatchResult:
    requirements: tuple[ArchitectureRequirementPatch, ...]
    architecture_cases: tuple[ArchitectureCasePatch, ...]
    architecture_export_offset: int | None
    architecture_metadata_offsets: tuple[int, ...]

    @property
    def ada_cases(self) -> tuple[ArchitectureCasePatch, ...]:
        """Return Ada patches for callers that use the former result field."""
        return tuple(
            patch
            for patch in self.architecture_cases
            if patch.architecture == ADA.name
        )


@dataclass(frozen=True)
class OutputPlan:
    output_path: Path
    backup_path: Path | None


def sha256_bytes(data: bytes | bytearray) -> str:
    return hashlib.sha256(data).hexdigest()


def pe_header_offset(data: bytes | bytearray) -> int:
    if len(data) < 0x40 or data[:2] != b"MZ":
        raise PatchError("The input is not a PE file.")
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if pe_offset + 24 > len(data) or data[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise PatchError("The input has an invalid PE header.")
    return int(pe_offset)


def read_pe_sections(data: bytes | bytearray) -> list[PeSection]:
    pe_offset = pe_header_offset(data)
    machine, section_count = struct.unpack_from("<HH", data, pe_offset + 4)
    if machine != PE_MACHINE_AMD64:
        raise PatchError(f"The input PE machine is 0x{machine:04x}, not AMD64.")
    optional_size = struct.unpack_from("<H", data, pe_offset + 20)[0]
    optional_offset = pe_offset + 24
    if optional_size < 2 or optional_offset + optional_size > len(data):
        raise PatchError("The input PE optional header is truncated.")
    optional_magic = struct.unpack_from("<H", data, optional_offset)[0]
    if optional_magic != PE32_PLUS_MAGIC:
        raise PatchError(
            f"The input optional-header magic is 0x{optional_magic:04x}, not PE32+."
        )
    section_table = optional_offset + optional_size
    if section_table + section_count * 40 > len(data):
        raise PatchError("The input PE section table is truncated.")
    sections = []
    for index in range(section_count):
        offset = section_table + index * 40
        virtual_size, virtual_address, raw_size, raw_offset = struct.unpack_from(
            "<IIII", data, offset + 8
        )
        if raw_size and (raw_offset > len(data) or raw_size > len(data) - raw_offset):
            raise PatchError(f"PE section {index} extends past the input file.")
        sections.append(PeSection(virtual_address, virtual_size, raw_offset, raw_size))
    return sections


def strip_authenticode(data: bytearray) -> AuthenticodeRemoval | None:
    """Remove the signature invalidated by patching while preserving other data."""
    sections = read_pe_sections(data)
    pe_offset = pe_header_offset(data)
    section_count = struct.unpack_from("<H", data, pe_offset + 6)[0]
    optional_size = struct.unpack_from("<H", data, pe_offset + 20)[0]
    optional_offset = pe_offset + 24

    # PE32+ stores NumberOfRvaAndSizes at +108 and its data directories at
    # +112. The security directory is entry 4. Its first value is a file
    # offset, unlike the RVAs in the other data-directory entries.
    directory_count_offset = optional_offset + 108
    directory_offset = optional_offset + 112 + 4 * 8
    optional_end = optional_offset + optional_size
    if directory_count_offset + 4 > optional_end:
        raise PatchError("The input PE optional header has no data directories.")
    directory_count = struct.unpack_from("<I", data, directory_count_offset)[0]
    if directory_count <= 4:
        return None
    if directory_offset + 8 > optional_end:
        raise PatchError("The PE security directory is truncated.")

    certificate_offset, certificate_size = struct.unpack_from(
        "<II", data, directory_offset
    )
    if not certificate_offset and not certificate_size:
        return None
    if not certificate_offset or not certificate_size:
        raise PatchError("The PE security directory is invalid.")
    if certificate_offset & 7:
        raise PatchError("The PE certificate table is not 8-byte aligned.")
    if (
        certificate_offset > len(data)
        or certificate_size > len(data) - certificate_offset
    ):
        raise PatchError("The PE certificate table extends past the input file.")

    # A certificate table is outside the mapped PE image. Reject a directory
    # that overlaps headers or section data so that removal cannot shift them.
    section_table_end = optional_end + section_count * 40
    size_of_headers = struct.unpack_from("<I", data, optional_offset + 60)[0]
    image_data_end = max(
        [
            section_table_end,
            size_of_headers,
            *(
                section.raw_offset + section.raw_size
                for section in sections
                if section.raw_size
            ),
        ]
    )
    if certificate_offset < image_data_end:
        raise PatchError("The PE certificate table overlaps PE image data.")

    certificate_end = certificate_offset + certificate_size
    certificate_count = 0
    position = certificate_offset
    while position < certificate_end:
        if certificate_end - position < 8:
            raise PatchError("The PE certificate table has a truncated entry header.")
        entry_size = struct.unpack_from("<I", data, position)[0]
        if entry_size < 8:
            raise PatchError("The PE certificate table has an invalid entry length.")
        aligned_size = (entry_size + 7) & ~7
        if aligned_size > certificate_end - position:
            raise PatchError("A PE certificate entry extends past its table.")
        position += aligned_size
        certificate_count += 1

    trailing_data_size = len(data) - certificate_end
    struct.pack_into("<II", data, directory_offset, 0, 0)
    del data[certificate_offset:certificate_end]
    return AuthenticodeRemoval(
        certificate_offset,
        certificate_size,
        certificate_count,
        trailing_data_size,
    )


def calculate_pe_checksum(data: bytes | bytearray) -> int:
    pe_offset = pe_header_offset(data)
    optional_offset = pe_offset + 24
    optional_size = struct.unpack_from("<H", data, pe_offset + 20)[0]
    if optional_size < 68 or optional_offset + optional_size > len(data):
        raise PatchError("The input PE optional header is truncated.")
    checksum_offset = optional_offset + 64

    checksum = 0
    padded_size = (len(data) + 3) & ~3
    for offset in range(0, padded_size, 4):
        if offset == checksum_offset:
            continue
        chunk = bytes(data[offset : offset + 4]).ljust(4, b"\0")
        checksum += struct.unpack("<I", chunk)[0]
        if checksum >= 1 << 32:
            checksum = (checksum & 0xFFFFFFFF) + (checksum >> 32)
    checksum = (checksum & 0xFFFF) + (checksum >> 16)
    checksum += checksum >> 16
    return (checksum & 0xFFFF) + len(data)


def update_pe_checksum(data: bytearray) -> int:
    pe_offset = pe_header_offset(data)
    checksum = calculate_pe_checksum(data)
    struct.pack_into("<I", data, pe_offset + 24 + 64, checksum)
    return checksum


def rva_to_file_offset(rva: int, sections: Sequence[PeSection]) -> int | None:
    for section in sections:
        size = max(section.virtual_size, section.raw_size)
        if section.virtual_address <= rva < section.virtual_address + size:
            raw = section.raw_offset + rva - section.virtual_address
            if raw < section.raw_offset + section.raw_size:
                return raw
    return None


def read_c_string(data: bytes | bytearray, offset: int, limit: int = 1024) -> str:
    if not 0 <= offset < len(data):
        return ""
    end = data.find(b"\0", offset, min(len(data), offset + limit))
    if end < 0:
        return ""
    return bytes(data[offset:end]).decode("ascii", errors="replace")


def read_pe_exports(
    data: bytes | bytearray, sections: Sequence[PeSection]
) -> dict[str, int]:
    """Return named PE exports as RVAs."""
    pe_offset = pe_header_offset(data)
    optional_offset = pe_offset + 24
    optional_size = struct.unpack_from("<H", data, pe_offset + 20)[0]
    # The PE32+ NumberOfRvaAndSizes and data-directory fields start here.
    if optional_size < 120 or optional_offset + optional_size > len(data):
        raise PatchError("The input PE optional header has no export directory.")
    if struct.unpack_from("<H", data, optional_offset)[0] != PE32_PLUS_MAGIC:
        raise PatchError("The input is not a PE32+ image.")
    directory_count = struct.unpack_from("<I", data, optional_offset + 108)[0]
    if directory_count < 1:
        return {}
    export_rva, export_size = struct.unpack_from("<II", data, optional_offset + 112)
    if not export_rva or not export_size:
        return {}
    export_offset = rva_to_file_offset(export_rva, sections)
    if export_offset is None or export_offset + 40 > len(data):
        raise PatchError("The PE export directory is outside the input file.")

    function_count, name_count, functions_rva, names_rva, ordinals_rva = (
        struct.unpack_from("<IIIII", data, export_offset + 20)
    )
    if function_count > 100_000 or name_count > 100_000:
        raise PatchError("The PE export table has an invalid entry count.")

    def table_offset(rva: int, size: int, label: str) -> int:
        offset = rva_to_file_offset(rva, sections)
        if offset is None or offset > len(data) or size > len(data) - offset:
            raise PatchError(f"The PE export {label} table is truncated.")
        return offset

    functions_offset = table_offset(functions_rva, function_count * 4, "address")
    names_offset = table_offset(names_rva, name_count * 4, "name")
    ordinals_offset = table_offset(ordinals_rva, name_count * 2, "ordinal")
    exports: dict[str, int] = {}
    for index in range(name_count):
        name_rva = struct.unpack_from("<I", data, names_offset + index * 4)[0]
        name_offset = rva_to_file_offset(name_rva, sections)
        if name_offset is None:
            raise PatchError("A PE export name is outside the input file.")
        name = read_c_string(data, name_offset)
        if not name:
            raise PatchError("A PE export name is invalid.")
        ordinal = struct.unpack_from("<H", data, ordinals_offset + index * 2)[0]
        if ordinal >= function_count:
            raise PatchError(f"PE export '{name}' has an invalid ordinal.")
        function_rva = struct.unpack_from("<I", data, functions_offset + ordinal * 4)[0]
        if name in exports:
            raise PatchError(f"The PE has duplicate export name '{name}'.")
        exports[name] = function_rva
    return exports


def find_fatbins(data: bytes | bytearray) -> list[FatbinLocation]:
    # Fatbins can appear anywhere in the PE data. Later code replaces each
    # container in place, so offsets outside a matched container stay stable.
    locations: list[FatbinLocation] = []
    search_at = 0
    covered_until = 0
    while True:
        offset = data.find(FATBIN_MAGIC_BYTES, search_at)
        if offset < 0:
            break
        search_at = offset + len(FATBIN_MAGIC_BYTES)
        if offset < covered_until or offset + 16 > len(data):
            continue
        magic, version, header_size, data_size = struct.unpack_from(
            "<IHHQ", data, offset
        )
        size = header_size + data_size
        if (
            magic != FATBIN_MAGIC
            or version != 1
            or header_size < 16
            or size < header_size
            or offset + size > len(data)
        ):
            continue
        locations.append(FatbinLocation(offset, header_size, data_size))
        covered_until = offset + size
    if not locations:
        raise PatchError("The input has no supported CUDA fatbins.")
    return locations


def parse_fatbin_records(blob: bytes | bytearray) -> list[FatbinRecord]:
    if len(blob) < 16:
        raise PatchError("A CUDA fatbin is too small.")
    magic, version, fat_header_size, data_size = struct.unpack_from("<IHHQ", blob, 0)
    if magic != FATBIN_MAGIC or version != 1:
        raise PatchError("A CUDA fatbin header is invalid.")
    end = fat_header_size + data_size
    if fat_header_size < 16 or end != len(blob):
        raise PatchError("A CUDA fatbin length does not match its header.")

    records: list[FatbinRecord] = []
    offset = fat_header_size
    while offset < end:
        if offset + 48 > end:
            raise PatchError("A CUDA fatbin record header is truncated.")
        kind, _version, header_size = struct.unpack_from("<HHI", blob, offset)
        payload_size = struct.unpack_from("<Q", blob, offset + 8)[0]
        flags = struct.unpack_from("<Q", blob, offset + 40)[0]
        next_offset = offset + header_size + payload_size
        if header_size < 48 or next_offset <= offset or next_offset > end:
            raise PatchError("A CUDA fatbin record has an invalid length.")
        records.append(FatbinRecord(offset, kind, header_size, payload_size, flags))
        offset = next_offset
    if offset != end:
        raise PatchError("A CUDA fatbin record extends past its container.")
    return records


def preserve_record_flags(generated: bytes, source: bytes) -> bytes:
    output = bytearray(generated)
    source_records = parse_fatbin_records(source)
    generated_records = parse_fatbin_records(output)
    source_low_flags: dict[int, int] = {}
    for record in source_records:
        source_low_flags.setdefault(record.kind, record.flags & 0xFF)
    for record in generated_records:
        if record.kind not in source_low_flags:
            continue
        flags = record.flags
        flags = (flags & ~0xFF) | source_low_flags[record.kind]
        struct.pack_into("<Q", output, record.offset + 40, flags)
    return bytes(output)


def tool_error(
    command: Sequence[os.PathLike[str] | str], result: subprocess.CompletedProcess[str]
) -> str:
    text = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    lines = text.splitlines()
    if len(lines) > 30:
        lines = lines[:10] + ["... output omitted ..."] + lines[-19:]
    rendered = " ".join(str(item) for item in command)
    detail = "\n".join(lines)
    return f"Command failed with exit code {result.returncode}: {rendered}\n{detail}".rstrip()


def run_tool(
    command: Sequence[os.PathLike[str] | str],
    *,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            [str(item) for item in command],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as error:
        rendered = " ".join(str(item) for item in command)
        raise PatchError(f"Cannot run command '{rendered}': {error}") from error
    if result.returncode:
        raise PatchError(tool_error(command, result))
    return result


def find_cuda_tool(name: str, cuda_bin: Path | None) -> Path:
    names = [name]
    if os.name == "nt":
        names.insert(0, f"{name}.exe")
    if cuda_bin is not None:
        for candidate_name in names:
            candidate = cuda_bin / candidate_name
            if candidate.is_file():
                return candidate.resolve()
        raise PatchError(f"CUDA tool '{name}' is not in {cuda_bin}.")
    for candidate_name in names:
        found = shutil.which(candidate_name)
        if found:
            return Path(found).resolve()
    raise PatchError(
        f"CUDA tool '{name}' is not on PATH. Install CUDA Toolkit 13.3 or use --cuda-bin."
    )


def find_cuda_tools(cuda_bin: Path | None) -> CudaTools:
    return CudaTools(
        cuobjdump=find_cuda_tool("cuobjdump", cuda_bin),
        ptxas=find_cuda_tool("ptxas", cuda_bin),
        fatbinary=find_cuda_tool("fatbinary", cuda_bin),
    )


def cuda_version(ptxas: Path) -> str:
    result = run_tool([ptxas, "--version"])
    text = f"{result.stdout}\n{result.stderr}"
    match = re.search(r"release\s+([0-9]+(?:\.[0-9]+)*)", text)
    return match.group(1) if match else "unknown"


def parse_listed_images(output: str, label: str) -> list[str]:
    pattern = re.compile(
        rf"^\s*{re.escape(label)}\s+file\s+\d+\s*:\s*(.+?)\s*$",
        re.IGNORECASE,
    )
    names = []
    for line in output.splitlines():
        match = pattern.match(line)
        if match:
            names.append(Path(match.group(1)).name)
    return names


def image_architecture(path: Path) -> int:
    match = re.search(r"\.sm_(\d+)[a-z]?\.(?:ptx|cubin)$", path.name, re.IGNORECASE)
    if not match:
        raise PatchError(f"Cannot read the CUDA architecture from '{path.name}'.")
    return int(match.group(1))


def extract_images(fatbin_path: Path, work_dir: Path, cuobjdump: Path) -> list[Image]:
    ptx_list = run_tool([cuobjdump, "--list-ptx", fatbin_path]).stdout
    elf_list = run_tool([cuobjdump, "--list-elf", fatbin_path]).stdout
    ptx_names = parse_listed_images(ptx_list, "PTX")
    elf_names = parse_listed_images(elf_list, "ELF")
    if not ptx_names or not elf_names:
        raise PatchError("A CUDA fatbin does not contain both PTX and ELF images.")

    run_tool([cuobjdump, "--extract-ptx", "all", fatbin_path], cwd=work_dir)
    run_tool([cuobjdump, "--extract-elf", "all", fatbin_path], cwd=work_dir)

    ptx_images = [
        Image("ptx", image_architecture(work_dir / name), work_dir / name)
        for name in ptx_names
    ]
    elf_images = [
        Image("elf", image_architecture(work_dir / name), work_dir / name)
        for name in elf_names
    ]
    for image in ptx_images + elf_images:
        if not image.path.is_file():
            raise PatchError(f"cuobjdump did not create '{image.path.name}'.")

    source_records = parse_fatbin_records(fatbin_path.read_bytes())
    ptx_index = 0
    elf_index = 0
    ordered: list[Image] = []
    for record in source_records:
        if record.kind == 1:
            if ptx_index >= len(ptx_images):
                raise PatchError(
                    "The PTX record count does not match cuobjdump output."
                )
            ordered.append(ptx_images[ptx_index])
            ptx_index += 1
        elif record.kind == 2:
            if elf_index >= len(elf_images):
                raise PatchError(
                    "The ELF record count does not match cuobjdump output."
                )
            ordered.append(elf_images[elf_index])
            elf_index += 1
        else:
            raise PatchError(f"Unsupported CUDA fatbin record kind {record.kind}.")
    if ptx_index != len(ptx_images) or elf_index != len(elf_images):
        raise PatchError("The CUDA image count does not match the fatbin records.")
    return ordered


def normalize_crlf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n").replace(b"\n", b"\r\n")


def canonical_ptx(data: bytes) -> bytes:
    """Normalize PTX line endings for comparison."""
    return re.sub(rb"\r+(?=\n)", b"", data)


def parse_ptx_target(text: str) -> int:
    matches = re.findall(r"(?m)^\s*\.target\s+sm_(\d+)[a-z]?\b", text)
    if len(matches) != 1:
        raise PatchError(f"Expected one PTX target directive, found {len(matches)}.")
    return int(matches[0])


def previous_mov_assignment(
    text: str, position: int, register: str, limit: int = 4000
) -> str:
    start = max(0, position - limit)
    pattern = re.compile(r"mov\.b32\s+" + re.escape(register) + r",\s*([^;]+?)\s*;")
    matches = list(pattern.finditer(text, start, position))
    if not matches:
        raise PatchError(f"Cannot find an assignment for PTX register {register}.")
    return matches[-1].group(1).strip()


def previous_literal_assignment(
    text: str, position: int, register: str, limit: int = 4000
) -> int:
    value = previous_mov_assignment(text, position, register, limit)
    if not re.fullmatch(r"\d+", value):
        raise PatchError(
            f"The previous assignment for PTX register {register} is not a literal."
        )
    return int(value)


PTX_PREDICATE = r"(?:%p\d+|[A-Za-z_][\w$]*)"

FP8_DOWN_PATTERN = re.compile(
    rf"(?P<guard>@!?{PTX_PREDICATE}\s+)?"
    r"cvt\.rn\.satfinite\.e4m3x2\.f16x2\s+"
    r"(?P<destination>%rs\d+),\s*(?P<source>%r\d+)\s*;"
)
FP8_UP_PATTERN = re.compile(
    rf"(?P<guard>@!?{PTX_PREDICATE}\s+)?"
    r"cvt\.rn\.f16x2\.e4m3x2\s+"
    r"(?P<destination>%r\d+),\s*(?P<source>%rs\d+)\s*;"
)
FP8_PACK_PATTERN = re.compile(
    rf"(?P<guard>@!?{PTX_PREDICATE}\s+)?"
    r"mov\.b32\s+(?P<destination>%r\d+),\s*"
    r"\{\s*(?P<low>%rs\d+),\s*(?P<high>%rs\d+)\s*\}\s*;"
)
FP8_MMA_PATTERN = re.compile(
    rf"(?P<guard>@!?{PTX_PREDICATE}\s+)?"
    r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s*"
    r"\{\s*(?P<d0>%r\d+),\s*(?P<d1>%r\d+)\s*\}\s*,\s*"
    r"\{\s*(?P<a0>%r\d+),\s*(?P<a1>%r\d+),\s*"
    r"(?P<a2>%r\d+),\s*(?P<a3>%r\d+)\s*\}\s*,\s*"
    r"\{\s*(?P<b0>%r\d+),\s*(?P<b1>%r\d+)\s*\}\s*,\s*"
    r"\{\s*(?P<c0>%r\d+),\s*(?P<c1>%r\d+)\s*\}\s*;"
)
PTX_ENTRY_PATTERN = re.compile(r"(?m)^\s*(?:\.visible\s+)?\.entry\b")


def ptx_register_count(text: str, register: str) -> int:
    return len(re.findall(re.escape(register) + r"(?!\d)", text))


def fp8_pair_to_half_lines(source: str, destination: str) -> list[str]:
    # Expand both E4M3 values exactly, including denormals, signed zero, and NaN.
    lines: list[str] = []
    for bit_offset in (0, 8):
        lines.extend(
            [
                f"bfe.u32 dlssnr_fp8_value, {source}, {bit_offset}, 8;",
                "and.b32 dlssnr_fp8_abs, dlssnr_fp8_value, 0x7f;",
                "and.b32 dlssnr_fp8_sign, dlssnr_fp8_value, 0x80;",
                "shl.b32 dlssnr_fp8_sign, dlssnr_fp8_sign, 8;",
                "add.u32 dlssnr_fp8_normal, dlssnr_fp8_abs, 0x40;",
                "shl.b32 dlssnr_fp8_normal, dlssnr_fp8_normal, 7;",
                "or.b32 dlssnr_fp8_normal, dlssnr_fp8_normal, "
                "dlssnr_fp8_sign;",
                "and.b32 dlssnr_fp8_mantissa, dlssnr_fp8_abs, 7;",
                "bfind.u32 dlssnr_fp8_msb, dlssnr_fp8_mantissa;",
                "sub.u32 dlssnr_fp8_shift, 10, dlssnr_fp8_msb;",
                "shl.b32 dlssnr_fp8_mantissa, dlssnr_fp8_mantissa, "
                "dlssnr_fp8_shift;",
                "and.b32 dlssnr_fp8_mantissa, dlssnr_fp8_mantissa, 0x3ff;",
                "add.u32 dlssnr_fp8_exponent, dlssnr_fp8_msb, 6;",
                "shl.b32 dlssnr_fp8_exponent, dlssnr_fp8_exponent, 10;",
                "or.b32 dlssnr_fp8_subnormal, dlssnr_fp8_sign, "
                "dlssnr_fp8_exponent;",
                "or.b32 dlssnr_fp8_subnormal, dlssnr_fp8_subnormal, "
                "dlssnr_fp8_mantissa;",
                "and.b32 dlssnr_fp8_exponent, dlssnr_fp8_abs, 0x78;",
                "setp.eq.u32 dlssnr_fp8_exp_zero, dlssnr_fp8_exponent, 0;",
                "setp.eq.u32 dlssnr_fp8_zero, dlssnr_fp8_abs, 0;",
                "setp.eq.u32 dlssnr_fp8_nan, dlssnr_fp8_abs, 0x7f;",
                "selp.b32 dlssnr_fp8_result, dlssnr_fp8_subnormal, "
                "dlssnr_fp8_normal, dlssnr_fp8_exp_zero;",
                "selp.b32 dlssnr_fp8_result, dlssnr_fp8_sign, "
                "dlssnr_fp8_result, dlssnr_fp8_zero;",
                "selp.b32 dlssnr_fp8_result, 0x7fff, "
                "dlssnr_fp8_result, dlssnr_fp8_nan;",
            ]
        )
        if bit_offset == 0:
            lines.append("mov.b32 dlssnr_fp8_low, dlssnr_fp8_result;")
        else:
            lines.extend(
                [
                    "shl.b32 dlssnr_fp8_result, dlssnr_fp8_result, 16;",
                    f"or.b32 {destination}, dlssnr_fp8_low, "
                    "dlssnr_fp8_result;",
                ]
            )
    return lines


def half_pair_to_fp8_lines(source: str, destination: str) -> list[str]:
    # Match cvt.rn.satfinite for both packed halves with integer RNE logic.
    lines = [f"mov.b32 dlssnr_f16_source, {source};"]
    for bit_offset in (0, 16):
        lines.extend(
            [
                f"bfe.u32 dlssnr_f16_value, dlssnr_f16_source, "
                f"{bit_offset}, 16;",
                "and.b32 dlssnr_f16_sign, dlssnr_f16_value, 0x8000;",
                "shr.u32 dlssnr_f16_sign, dlssnr_f16_sign, 8;",
                "and.b32 dlssnr_f16_abs, dlssnr_f16_value, 0x7fff;",
                "shr.u32 dlssnr_f16_normal, dlssnr_f16_abs, 7;",
                "sub.u32 dlssnr_f16_normal, dlssnr_f16_normal, 0x40;",
                "and.b32 dlssnr_f16_remainder, dlssnr_f16_abs, 0x7f;",
                "and.b32 dlssnr_f16_lsb, dlssnr_f16_normal, 1;",
                "setp.gt.u32 dlssnr_f16_gt, dlssnr_f16_remainder, 0x40;",
                "setp.eq.u32 dlssnr_f16_eq, dlssnr_f16_remainder, 0x40;",
                "setp.ne.u32 dlssnr_f16_odd, dlssnr_f16_lsb, 0;",
                "and.pred dlssnr_f16_tie, dlssnr_f16_eq, dlssnr_f16_odd;",
                "or.pred dlssnr_f16_round, dlssnr_f16_gt, dlssnr_f16_tie;",
                "selp.b32 dlssnr_f16_increment, 1, 0, dlssnr_f16_round;",
                "add.u32 dlssnr_f16_normal, dlssnr_f16_normal, "
                "dlssnr_f16_increment;",
                "shr.u32 dlssnr_f16_exponent, dlssnr_f16_abs, 10;",
                "and.b32 dlssnr_f16_exponent, dlssnr_f16_exponent, 0x1f;",
                "and.b32 dlssnr_f16_mantissa, dlssnr_f16_abs, 0x3ff;",
                "or.b32 dlssnr_f16_mantissa, dlssnr_f16_mantissa, 0x400;",
                "sub.u32 dlssnr_f16_shift, 16, dlssnr_f16_exponent;",
                "min.u32 dlssnr_f16_shift, dlssnr_f16_shift, 31;",
                "shr.u32 dlssnr_f16_subnormal, dlssnr_f16_mantissa, "
                "dlssnr_f16_shift;",
                "shl.b32 dlssnr_f16_scale, 1, dlssnr_f16_shift;",
                "sub.u32 dlssnr_f16_mask, dlssnr_f16_scale, 1;",
                "and.b32 dlssnr_f16_remainder, dlssnr_f16_mantissa, "
                "dlssnr_f16_mask;",
                "shr.u32 dlssnr_f16_half, dlssnr_f16_scale, 1;",
                "and.b32 dlssnr_f16_lsb, dlssnr_f16_subnormal, 1;",
                "setp.gt.u32 dlssnr_f16_gt, dlssnr_f16_remainder, "
                "dlssnr_f16_half;",
                "setp.eq.u32 dlssnr_f16_eq, dlssnr_f16_remainder, "
                "dlssnr_f16_half;",
                "setp.ne.u32 dlssnr_f16_odd, dlssnr_f16_lsb, 0;",
                "and.pred dlssnr_f16_tie, dlssnr_f16_eq, dlssnr_f16_odd;",
                "or.pred dlssnr_f16_round, dlssnr_f16_gt, dlssnr_f16_tie;",
                "selp.b32 dlssnr_f16_increment, 1, 0, dlssnr_f16_round;",
                "add.u32 dlssnr_f16_subnormal, dlssnr_f16_subnormal, "
                "dlssnr_f16_increment;",
                "setp.lt.u32 dlssnr_f16_denormal, dlssnr_f16_abs, 0x2400;",
                "setp.le.u32 dlssnr_f16_underflow, dlssnr_f16_abs, 0x1400;",
                "setp.gt.u32 dlssnr_f16_overflow, dlssnr_f16_abs, 0x5f40;",
                "setp.gt.u32 dlssnr_f16_nan, dlssnr_f16_abs, 0x7c00;",
                "selp.b32 dlssnr_f16_result, dlssnr_f16_subnormal, "
                "dlssnr_f16_normal, dlssnr_f16_denormal;",
                "selp.b32 dlssnr_f16_result, 0, dlssnr_f16_result, "
                "dlssnr_f16_underflow;",
                "selp.b32 dlssnr_f16_result, 0x7e, dlssnr_f16_result, "
                "dlssnr_f16_overflow;",
                "or.b32 dlssnr_f16_result, dlssnr_f16_result, "
                "dlssnr_f16_sign;",
                "selp.b32 dlssnr_f16_result, 0x7f, dlssnr_f16_result, "
                "dlssnr_f16_nan;",
            ]
        )
        if bit_offset == 0:
            lines.append("mov.b32 dlssnr_f16_low, dlssnr_f16_result;")
        else:
            lines.extend(
                [
                    "shl.b32 dlssnr_f16_result, dlssnr_f16_result, 8;",
                    "or.b32 dlssnr_f16_result, dlssnr_f16_low, "
                    "dlssnr_f16_result;",
                    f"cvt.u16.u32 {destination}, dlssnr_f16_result;",
                ]
            )
    return lines


def fp8_conversion_declarations() -> list[str]:
    return [
        ".reg .b32 dlssnr_fp8_value, dlssnr_fp8_abs, dlssnr_fp8_sign;",
        ".reg .b32 dlssnr_fp8_normal, dlssnr_fp8_mantissa;",
        ".reg .b32 dlssnr_fp8_msb, dlssnr_fp8_shift, dlssnr_fp8_exponent;",
        ".reg .b32 dlssnr_fp8_subnormal, dlssnr_fp8_result, dlssnr_fp8_low;",
        ".reg .pred dlssnr_fp8_exp_zero, dlssnr_fp8_zero, dlssnr_fp8_nan;",
    ]


def half_conversion_declarations() -> list[str]:
    return [
        ".reg .b32 dlssnr_f16_source, dlssnr_f16_value, dlssnr_f16_abs;",
        ".reg .b32 dlssnr_f16_sign, dlssnr_f16_normal, dlssnr_f16_remainder;",
        ".reg .b32 dlssnr_f16_lsb, dlssnr_f16_increment, dlssnr_f16_exponent;",
        ".reg .b32 dlssnr_f16_mantissa, dlssnr_f16_shift, dlssnr_f16_scale;",
        ".reg .b32 dlssnr_f16_mask, dlssnr_f16_half, dlssnr_f16_subnormal;",
        ".reg .b32 dlssnr_f16_result, dlssnr_f16_low;",
        ".reg .pred dlssnr_f16_gt, dlssnr_f16_eq, dlssnr_f16_odd;",
        ".reg .pred dlssnr_f16_tie, dlssnr_f16_round;",
        ".reg .pred dlssnr_f16_denormal, dlssnr_f16_underflow;",
        ".reg .pred dlssnr_f16_overflow, dlssnr_f16_nan;",
    ]


def fp8_pack_sources(text: str) -> dict[str, tuple[str, str, int]]:
    down_sources: dict[str, list[str]] = {}
    for match in FP8_DOWN_PATTERN.finditer(text):
        down_sources.setdefault(match.group("destination"), []).append(
            match.group("source")
        )

    packed_sources: dict[str, list[tuple[str, str, int]]] = {}
    for match in FP8_PACK_PATTERN.finditer(text):
        low_sources = down_sources.get(match.group("low"), [])
        high_sources = down_sources.get(match.group("high"), [])
        if len(low_sources) == len(high_sources) == 1:
            packed_sources.setdefault(match.group("destination"), []).append(
                (low_sources[0], high_sources[0], match.start())
            )
    return {
        register: sources[0]
        for register, sources in packed_sources.items()
        if len(sources) == 1
    }


def render_fp8_mma_run(
    matches: Sequence[re.Match[str]],
    packed_sources: dict[str, tuple[str, str, int]],
) -> str:
    operands: list[str] = []
    for match in matches:
        for name in ("a0", "a1", "a2", "a3", "b0", "b1"):
            register = match.group(name)
            if register not in operands:
                operands.append(register)
    names = {
        register: (f"dlssnr_fp16_{index}_0", f"dlssnr_fp16_{index}_1")
        for index, register in enumerate(operands)
    }
    mapped = {
        register: sources
        for register, sources in packed_sources.items()
        if sources[2] < matches[0].start()
    }
    # FP8 k32 and FP16 k16 distribute each fragment differently across a
    # four-lane group. Each source register becomes two shuffled FP16 pairs;
    # two k16 operations then cover the original k32 accumulation.
    has_encoded_operands = any(register not in mapped for register in operands)

    lines = ["{"]
    lines.extend(
        [
            ".reg .u32 dlssnr_lane, dlssnr_group, dlssnr_pair;",
            ".reg .u32 dlssnr_source0, dlssnr_source1, dlssnr_parity;",
            ".reg .b32 dlssnr_shuffle0, dlssnr_shuffle1, dlssnr_shifted;",
            ".reg .b32 dlssnr_selected;",
            ".reg .pred dlssnr_high;",
            ".reg .b32 "
            + ", ".join(name for pair in names.values() for name in pair)
            + ";",
        ]
    )
    if has_encoded_operands:
        lines.extend(fp8_conversion_declarations())
    lines.extend(
        [
            "mov.u32 dlssnr_lane, %laneid;",
            "and.b32 dlssnr_group, dlssnr_lane, 0x1c;",
            "shr.u32 dlssnr_pair, dlssnr_lane, 1;",
            "and.b32 dlssnr_pair, dlssnr_pair, 1;",
            "add.u32 dlssnr_source0, dlssnr_group, dlssnr_pair;",
            "add.u32 dlssnr_source1, dlssnr_source0, 2;",
            "and.b32 dlssnr_parity, dlssnr_lane, 1;",
            "setp.ne.u32 dlssnr_high, dlssnr_parity, 0;",
        ]
    )

    for register in operands:
        first, second = names[register]
        if register in mapped:
            low, high, _position = mapped[register]
            for source_index, destination in (
                ("dlssnr_source0", first),
                ("dlssnr_source1", second),
            ):
                lines.extend(
                    [
                        "shfl.sync.idx.b32 dlssnr_shuffle0, "
                        f"{low}, {source_index}, 0x1f, 0xffffffff;",
                        "shfl.sync.idx.b32 dlssnr_shuffle1, "
                        f"{high}, {source_index}, 0x1f, 0xffffffff;",
                        f"selp.b32 {destination}, dlssnr_shuffle1, "
                        "dlssnr_shuffle0, dlssnr_high;",
                    ]
                )
        else:
            for source_index, destination in (
                ("dlssnr_source0", first),
                ("dlssnr_source1", second),
            ):
                lines.extend(
                    [
                        "shfl.sync.idx.b32 dlssnr_shuffle0, "
                        f"{register}, {source_index}, 0x1f, 0xffffffff;",
                        "shr.u32 dlssnr_shifted, dlssnr_shuffle0, 16;",
                        "selp.b32 dlssnr_selected, dlssnr_shifted, "
                        "dlssnr_shuffle0, dlssnr_high;",
                    ]
                )
                lines.extend(fp8_pair_to_half_lines("dlssnr_selected", destination))

    for match in matches:
        d0, d1 = match.group("d0"), match.group("d1")
        c0, c1 = match.group("c0"), match.group("c1")
        a0, a1 = names[match.group("a0")]
        a2, a3 = names[match.group("a1")]
        a4, a5 = names[match.group("a2")]
        a6, a7 = names[match.group("a3")]
        b0, b1 = names[match.group("b0")]
        b2, b3 = names[match.group("b1")]
        lines.extend(
            [
                "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                f"{{{d0}, {d1}}}, {{{a0}, {a2}, {a1}, {a3}}}, "
                f"{{{b0}, {b1}}}, {{{c0}, {c1}}};",
                "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                f"{{{d0}, {d1}}}, {{{a4}, {a6}, {a5}, {a7}}}, "
                f"{{{b2}, {b3}}}, {{{d0}, {d1}}};",
            ]
        )
    lines.append("}")
    return "\n".join(lines)


def lower_fp8_mma_entry(text: str) -> tuple[str, int]:
    matches = list(FP8_MMA_PATTERN.finditer(text))
    if not matches:
        return text, 0
    runs: list[list[re.Match[str]]] = [[matches[0]]]
    for match in matches[1:]:
        if text[runs[-1][-1].end() : match.start()].strip():
            runs.append([match])
        else:
            runs[-1].append(match)

    packed_sources = fp8_pack_sources(text)
    replacements = [
        (
            run[0].start(),
            run[-1].end(),
            render_fp8_mma_run(run, packed_sources),
        )
        for run in runs
    ]
    for start, end, replacement in reversed(replacements):
        text = text[:start] + replacement + text[end:]

    text = FP8_PACK_PATTERN.sub(
        lambda match: ""
        if ptx_register_count(text, match.group("destination")) == 1
        else match.group(0),
        text,
    )
    text = FP8_DOWN_PATTERN.sub(
        lambda match: ""
        if ptx_register_count(text, match.group("destination")) == 1
        else match.group(0),
        text,
    )
    return text, len(matches)


def lower_fp8_mma(text: str) -> tuple[str, int]:
    starts = [match.start() for match in PTX_ENTRY_PATTERN.finditer(text)]
    if not starts:
        return lower_fp8_mma_entry(text)
    boundaries = [0, *starts, len(text)]
    parts: list[str] = []
    count = 0
    for start, end in zip(boundaries, boundaries[1:]):
        part, part_count = lower_fp8_mma_entry(text[start:end])
        parts.append(part)
        count += part_count
    return "".join(parts), count


def lower_fp8_conversions(text: str, stats: TransformStats) -> str:
    generic_down_count = len(
        re.findall(r"\bcvt\.[^;\n]*\.e4m3x2\.f16x2\b", text)
    )
    generic_up_count = len(
        re.findall(r"\bcvt\.[^;\n]*\.f16x2\.e4m3x2\b", text)
    )
    down_count = len(FP8_DOWN_PATTERN.findall(text))
    up_count = len(FP8_UP_PATTERN.findall(text))
    if down_count != generic_down_count:
        raise PatchError("An FP16-to-FP8 conversion has an unsupported form.")
    if up_count != generic_up_count:
        raise PatchError("An FP8-to-FP16 conversion has an unsupported form.")

    def replace_down(match: re.Match[str]) -> str:
        lines = ["{", *half_conversion_declarations()]
        lines.extend(
            half_pair_to_fp8_lines(
                match.group("source"), match.group("destination")
            )
        )
        lines.append("}")
        return "\n".join(lines)

    def replace_up(match: re.Match[str]) -> str:
        lines = ["{", ".reg .b32 dlssnr_fp8_source;"]
        lines.extend(fp8_conversion_declarations())
        lines.append(
            f"cvt.u32.u16 dlssnr_fp8_source, {match.group('source')};"
        )
        lines.extend(
            fp8_pair_to_half_lines(
                "dlssnr_fp8_source", match.group("destination")
            )
        )
        lines.append("}")
        return "\n".join(lines)

    text, down_count = FP8_DOWN_PATTERN.subn(replace_down, text)
    text, up_count = FP8_UP_PATTERN.subn(replace_up, text)
    stats.add("fp16_to_fp8", down_count)
    stats.add("fp8_to_fp16", up_count)
    return text


def lower_fp8_operations(text: str, stats: TransformStats) -> str:
    # Pre-Ada GPUs execute the source E4M3 path through integer conversions and
    # FP16 MMAs, so results and performance can differ from native FP8.
    generic_mma_count = len(
        re.findall(
            r"\bmma\.[^;]*\.f16\.e4m3\.e4m3\.f16\b", text, re.DOTALL
        )
    )
    mma_matches = list(FP8_MMA_PATTERN.finditer(text))
    if len(mma_matches) != generic_mma_count:
        raise PatchError("An FP8 matrix operation has an unsupported form.")
    if any(match.group("guard") for match in mma_matches):
        raise PatchError("A predicated FP8 matrix operation is unsupported.")
    if any(match.group("guard") for match in FP8_DOWN_PATTERN.finditer(text)):
        raise PatchError("A predicated FP16-to-FP8 conversion is unsupported.")
    if any(match.group("guard") for match in FP8_UP_PATTERN.finditer(text)):
        raise PatchError("A predicated FP8-to-FP16 conversion is unsupported.")
    if any(match.group("guard") for match in FP8_PACK_PATTERN.finditer(text)):
        raise PatchError("A predicated FP8 pack operation is unsupported.")
    text, mma_count = lower_fp8_mma(text)
    stats.add("fp8_mma", mma_count)
    return lower_fp8_conversions(text, stats)


F16_MMA_REGISTER = r"(?:%r\d+|dlssnr_fp16_\d+_[01])"
F16_MMA_PATTERN = re.compile(
    rf"(?P<guard>@!?{PTX_PREDICATE}\s+)?"
    r"mma\.sync\.aligned\.m16n8k16\.row\.col\.f16\.f16\.f16\.f16\s*"
    r"\{\s*(?P<d0>%r\d+),\s*(?P<d1>%r\d+)\s*\}\s*,\s*"
    rf"\{{\s*(?P<a0>{F16_MMA_REGISTER}),\s*"
    rf"(?P<a1>{F16_MMA_REGISTER}),\s*"
    rf"(?P<a2>{F16_MMA_REGISTER}),\s*"
    rf"(?P<a3>{F16_MMA_REGISTER})\s*\}}\s*,\s*"
    rf"\{{\s*(?P<b0>{F16_MMA_REGISTER}),\s*"
    rf"(?P<b1>{F16_MMA_REGISTER})\s*\}}\s*,\s*"
    r"\{\s*(?P<c0>%r\d+),\s*(?P<c1>%r\d+)\s*\}\s*;"
)
F16_MIN_MAX_PATTERN = re.compile(
    rf"(?P<guard>@!?{PTX_PREDICATE}\s+)?"
    r"(?P<operation>min|max)\.f16x2\s+"
    r"(?P<destination>%r\d+),\s*(?P<first>%r\d+),\s*"
    r"(?P<second>%r\d+)\s*;"
)
SIMPLE_ASYNC_COPY_PATTERN = re.compile(
    rf"(?P<guard>@!?{PTX_PREDICATE}\s+)?"
    r"cp\.async\.(?P<cache>ca|cg)\.shared\.global\s+"
    r"\[(?P<destination>%r\d+)\],\s*\[(?P<source>%rd\d+)\],\s*"
    r"(?P<size>4|8|16)\s*;"
)
ASYNC_COMMIT_PATTERN = re.compile(
    rf"(?P<guard>@!?{PTX_PREDICATE}\s+)?cp\.async\.commit_group\s*;"
)
ASYNC_WAIT_PATTERN = re.compile(
    rf"(?P<guard>@!?{PTX_PREDICATE}\s+)?cp\.async\.wait_group\s+0\s*;"
)
MBARRIER_INIT_PATTERN = re.compile(
    rf"(?P<guard>@!?{PTX_PREDICATE}\s+)?"
    r"mbarrier\.init\.shared\.b64\s+"
    r"\[(?P<storage>%r\d+)\],\s*(?P<count>%r\d+)\s*;"
)
MBARRIER_ARRIVE_PATTERN = re.compile(
    rf"(?P<guard>@!?{PTX_PREDICATE}\s+)?"
    r"mbarrier\.arrive\.shared::cta\.b64\s+"
    r"(?P<state>%rd\d+),\s*\[(?P<storage>%r\d+)\]\s*;"
)
MBARRIER_WAIT_PATTERN = re.compile(
    rf"(?P<guard>@!?{PTX_PREDICATE}\s+)?"
    r"mbarrier\.test_wait\.shared::cta\.b64\s+"
    rf"(?P<result>{PTX_PREDICATE}),\s*"
    r"\[(?P<storage>%r\d+)\],\s*(?P<state>%rd\d+)\s*;"
)


def lower_turing_mma(text: str, stats: TransformStats) -> str:
    generic_count = len(
        re.findall(
            r"\bmma\.[^;]*\.m16n8k16\.[^;]*\.f16\.f16\.f16\.f16\b",
            text,
            re.DOTALL,
        )
    )
    matches = list(F16_MMA_PATTERN.finditer(text))
    if len(matches) != generic_count:
        raise PatchError("An FP16 matrix operation has an unsupported form.")
    if any(match.group("guard") for match in matches):
        raise PatchError("A predicated FP16 matrix operation is unsupported for Turing.")

    def replace(match: re.Match[str]) -> str:
        d0, d1 = match.group("d0"), match.group("d1")
        c0, c1 = match.group("c0"), match.group("c1")
        return "\n".join(
            [
                "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                f"{{{d0}, {d1}}}, "
                f"{{{match.group('a0')}, {match.group('a1')}}}, "
                f"{{{match.group('b0')}}}, {{{c0}, {c1}}};",
                "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                f"{{{d0}, {d1}}}, "
                f"{{{match.group('a2')}, {match.group('a3')}}}, "
                f"{{{match.group('b1')}}}, {{{d0}, {d1}}};",
            ]
        )

    text, count = F16_MMA_PATTERN.subn(replace, text)
    stats.add("mma_k16", count)
    return text


def lower_turing_half_min_max(text: str, stats: TransformStats) -> str:
    generic_count = len(re.findall(r"\b(?:min|max)\.f16(?:x2)?\b", text))
    matches = list(F16_MIN_MAX_PATTERN.finditer(text))
    if len(matches) != generic_count:
        raise PatchError("A half-precision minimum or maximum has an unsupported form.")
    if any(match.group("guard") for match in matches):
        raise PatchError(
            "A predicated half-precision minimum or maximum is unsupported for Turing."
        )

    def replace(match: re.Match[str]) -> str:
        operation = match.group("operation")
        lines = [
            "{",
            ".reg .b16 dlssnr_half_a0, dlssnr_half_a1;",
            ".reg .b16 dlssnr_half_b0, dlssnr_half_b1;",
            ".reg .b16 dlssnr_half_d0, dlssnr_half_d1;",
            ".reg .f32 dlssnr_half_a, dlssnr_half_b, dlssnr_half_result;",
            f"mov.b32 {{dlssnr_half_a0, dlssnr_half_a1}}, "
            f"{match.group('first')};",
            f"mov.b32 {{dlssnr_half_b0, dlssnr_half_b1}}, "
            f"{match.group('second')};",
        ]
        for suffix in ("0", "1"):
            lines.extend(
                [
                    f"cvt.f32.f16 dlssnr_half_a, dlssnr_half_a{suffix};",
                    f"cvt.f32.f16 dlssnr_half_b, dlssnr_half_b{suffix};",
                    f"{operation}.f32 dlssnr_half_result, "
                    "dlssnr_half_a, dlssnr_half_b;",
                    f"cvt.rn.f16.f32 dlssnr_half_d{suffix}, "
                    "dlssnr_half_result;",
                ]
            )
        lines.extend(
            [
                f"mov.b32 {match.group('destination')}, "
                "{dlssnr_half_d0, dlssnr_half_d1};",
                "}",
            ]
        )
        return "\n".join(lines)

    operation_counts = {
        operation: len(
            re.findall(rf"\b{operation}\.f16x2\b", text)
        )
        for operation in ("min", "max")
    }
    text = F16_MIN_MAX_PATTERN.sub(replace, text)
    for operation, count in operation_counts.items():
        stats.add(f"half_{operation}", count)
    return text


def lower_turing_async_copies(text: str, stats: TransformStats) -> str:
    generic_count = len(re.findall(r"\bcp\.async(?:\.[^;\n]*)?\s*[^;\n]*;", text))
    copy_matches = list(SIMPLE_ASYNC_COPY_PATTERN.finditer(text))
    commit_matches = list(ASYNC_COMMIT_PATTERN.finditer(text))
    wait_matches = list(ASYNC_WAIT_PATTERN.finditer(text))
    copy_count = len(copy_matches)
    commit_count = len(commit_matches)
    wait_count = len(wait_matches)
    if copy_count + commit_count + wait_count != generic_count:
        raise PatchError("An asynchronous copy has an unsupported form for Turing.")
    if any(
        match.group("guard")
        for matches in (copy_matches, commit_matches, wait_matches)
        for match in matches
    ):
        raise PatchError("A predicated asynchronous copy is unsupported for Turing.")
    if commit_count != wait_count:
        raise PatchError("Turing asynchronous copy groups are not paired.")

    def replace(match: re.Match[str]) -> str:
        size = int(match.group("size"))
        register_count = size // 4
        registers = [f"dlssnr_copy{index}" for index in range(register_count)]
        if register_count == 1:
            register_type = ".b32"
            operand = registers[0]
        else:
            register_type = f".v{register_count}.b32"
            operand = "{" + ", ".join(registers) + "}"
        return "\n".join(
            [
                "{",
                ".reg .b32 " + ", ".join(registers) + ";",
                f"ld.global.{match.group('cache')}{register_type} {operand}, "
                f"[{match.group('source')}];",
                f"st.shared{register_type} [{match.group('destination')}], "
                f"{operand};",
                "}",
            ]
        )

    text, count = SIMPLE_ASYNC_COPY_PATTERN.subn(replace, text)
    text = ASYNC_COMMIT_PATTERN.sub("", text)
    text = ASYNC_WAIT_PATTERN.sub("", text)
    stats.add("async_copy", count)
    stats.add("async_group", commit_count)
    return text


def lower_turing_mbarriers(text: str, stats: TransformStats) -> str:
    generic_count = len(re.findall(r"\bmbarrier\.[^;\n]+;", text))
    init_matches = list(MBARRIER_INIT_PATTERN.finditer(text))
    arrive_matches = list(MBARRIER_ARRIVE_PATTERN.finditer(text))
    wait_matches = list(MBARRIER_WAIT_PATTERN.finditer(text))
    init_count = len(init_matches)
    arrive_count = len(arrive_matches)
    wait_count = len(wait_matches)
    if init_count + arrive_count + wait_count != generic_count:
        raise PatchError("An mbarrier operation has an unsupported form for Turing.")
    if any(
        match.group("guard")
        for matches in (init_matches, arrive_matches, wait_matches)
        for match in matches
    ):
        raise PatchError("A predicated mbarrier operation is unsupported for Turing.")
    if arrive_count != wait_count:
        raise PatchError("Turing mbarrier arrivals and waits are not paired.")

    # A full-CTA barrier is equivalent only when the source mbarrier expects one
    # arrival from every thread in the block.
    for match in init_matches:
        start = max(0, match.start() - 2000)
        prefix = text[start : match.start()]
        count_register = match.group("count")
        multiply_pattern = re.compile(
            r"mul\.lo\.s32\s+"
            + re.escape(count_register)
            + r",\s*(%r\d+),\s*(%r\d+)\s*;"
        )
        multiply_matches = list(multiply_pattern.finditer(prefix))
        if not multiply_matches:
            raise PatchError("A Turing mbarrier does not count all block threads.")
        multiply = multiply_matches[-1]
        dimensions = []
        for register in multiply.groups():
            assignment_pattern = re.compile(
                r"mov\.u32\s+"
                + re.escape(register)
                + r",\s*(%ntid\.[xyz])\s*;"
            )
            assignments = list(
                assignment_pattern.finditer(prefix, 0, multiply.start())
            )
            if not assignments:
                raise PatchError("A Turing mbarrier has an unknown arrival count.")
            dimensions.append(assignments[-1].group(1))
        if dimensions != ["%ntid.x", "%ntid.y"]:
            raise PatchError("A Turing mbarrier does not count all block threads.")

    text = MBARRIER_INIT_PATTERN.sub("", text)
    text = MBARRIER_ARRIVE_PATTERN.sub("bar.sync 0;", text)
    text = MBARRIER_WAIT_PATTERN.sub(r"mov.pred \g<result>, 1;", text)
    stats.add("barrier_init", init_count)
    stats.add("barrier_sync", arrive_count)
    return text


def lower_turing_operations(text: str, stats: TransformStats) -> str:
    text = lower_turing_mma(text, stats)
    text = lower_turing_half_min_max(text, stats)
    text = lower_turing_async_copies(text, stats)
    return lower_turing_mbarriers(text, stats)


def transform_ptx(
    source: str, target: Architecture = ADA
) -> tuple[str, TransformStats]:
    # Keep these conversions narrow. If an instruction has an unknown form,
    # fail instead of producing PTX that may run incorrectly on the target.
    stats = TransformStats()

    # Limit the PTX version before ptxas sees it; the selected toolchain uses
    # PTX 9.3 for the pre-Blackwell targets supported by these conversions.
    version_pattern = re.compile(r"(?m)^(\s*\.version\s+)(\d+)\.(\d+)(\s*)$")
    version_matches = list(version_pattern.finditer(source))
    if len(version_matches) != 1:
        raise PatchError(
            f"Expected one PTX version directive, found {len(version_matches)}."
        )

    def replace_version(match: re.Match[str]) -> str:
        version = (int(match.group(2)), int(match.group(3)))
        if (
            target.cuda >= BLACKWELL.cuda
            or version <= MAX_COMPATIBLE_PTX_VERSION
        ):
            return match.group(0)
        stats.add("ptx_version", 1)
        return (
            f"{match.group(1)}{MAX_COMPATIBLE_PTX_VERSION[0]}."
            f"{MAX_COMPATIBLE_PTX_VERSION[1]}{match.group(4)}"
        )

    output = version_pattern.sub(replace_version, source)

    target_pattern = re.compile(r"(?m)^(\s*\.target\s+)sm_\d+[a-z]?(.*)$")
    target_matches = list(target_pattern.finditer(output))
    if len(target_matches) != 1:
        raise PatchError(
            f"Expected one PTX target directive, found {len(target_matches)}."
        )
    output = target_pattern.sub(rf"\g<1>{target.cuda_name}\2", output)
    stats.add("target", 1)

    if target.cuda >= BLACKWELL.cuda:
        return output, stats
    if target.cuda < ADA.cuda:
        output = lower_fp8_operations(output, stats)

    # Pre-Blackwell targets have no equivalent for the newer warp bulk-copy
    # instruction. Accept only ordered elect/copy/expect groups whose barrier
    # operands match.
    elect_pattern = re.compile(
        rf"elect\.sync\s+_\|({PTX_PREDICATE}),\s*%r\d+\s*;"
    )
    bulk_pattern = re.compile(
        rf"(?P<guard>@!?{PTX_PREDICATE}\s+)?"
        r"cp\.async\.bulk\.shared::cta\.global\.mbarrier::complete_tx::bytes\s+"
        r"\[(?P<destination>%r\d+)\],\s*"
        r"\[(?P<source>%rd\d+)\],\s*(?P<size>%r\d+),\s*"
        r"\[(?P<barrier>%r\d+)\]\s*;"
    )
    expect_pattern = re.compile(
        rf"(?P<guard>@!?{PTX_PREDICATE}\s+)?"
        r"mbarrier\.expect_tx(?:\.[A-Za-z0-9_:]+)*\s+"
        r"\[(?P<barrier>%r\d+)\],\s*(?P<size>%r\d+)\s*;"
    )
    elect_matches = list(elect_pattern.finditer(output))
    bulk_matches = list(bulk_pattern.finditer(output))
    expect_matches = list(expect_pattern.finditer(output))
    generic_elects = list(re.finditer(r"\belect\.sync\b", output))
    generic_bulks = list(re.finditer(r"\bcp\.async\.bulk\b", output))
    generic_expects = list(re.finditer(r"\bmbarrier\.expect_tx\b", output))
    if len(elect_matches) != len(generic_elects):
        raise PatchError("An elect operation has an unsupported form.")
    if len(bulk_matches) != len(generic_bulks):
        raise PatchError("A bulk-copy operation has an unsupported form.")
    if any(match.group("guard") for match in bulk_matches):
        raise PatchError("A predicated bulk-copy operation is unsupported.")
    if len(expect_matches) != len(generic_expects):
        raise PatchError("A transaction expectation has an unsupported form.")
    if any(match.group("guard") for match in expect_matches):
        raise PatchError("A predicated transaction expectation is unsupported.")

    ordered_copy_operations = sorted(
        [(match.start(), "elect") for match in elect_matches]
        + [(match.start(), "bulk") for match in bulk_matches]
        + [(match.start(), "expect") for match in expect_matches]
    )
    expected_copy_order = ["elect", "bulk", "expect"] * len(bulk_matches)
    if [name for _position, name in ordered_copy_operations] != expected_copy_order:
        raise PatchError(
            "PTX elect, bulk-copy, and transaction-expectation operations are not "
            "ordered in supported groups."
        )
    for elect, bulk, expect in zip(
        elect_matches, bulk_matches, expect_matches, strict=True
    ):
        if elect.group(1) not in output[elect.end() : bulk.start()]:
            raise PatchError("An elected predicate does not control its bulk copy.")
        if (bulk.group("barrier"), bulk.group("size")) != (
            expect.group("barrier"),
            expect.group("size"),
        ):
            raise PatchError(
                "A bulk copy and its transaction expectation do not match."
            )

    # Arrival and wait operations must also form ordered pairs for the same
    # barrier state. Generated PTX can calculate the same storage in different
    # registers, so the state token is the stable relationship.
    arrive_pattern = re.compile(
        r"mbarrier\.arrive\.shared::cta\.b64\s+"
        r"(%rd\d+),\s*\[(%r\d+)\],\s*(%r\d+)\s*;"
    )
    wait_pattern = re.compile(
        rf"mbarrier\.try_wait\.shared::cta\.b64\s+"
        rf"({PTX_PREDICATE}),\s*\[(%r\d+)\],\s*(%rd\d+)\s*;"
    )
    arrive_matches = list(arrive_pattern.finditer(output))
    wait_matches = list(wait_pattern.finditer(output))
    generic_arrivals = list(
        re.finditer(r"\bmbarrier\.arrive\.shared::cta\.b64\b", output)
    )
    generic_waits = list(
        re.finditer(r"\bmbarrier\.try_wait\.shared::cta\.b64\b", output)
    )
    if len(arrive_matches) != len(generic_arrivals):
        raise PatchError("A barrier-arrival operation has an unsupported form.")
    if len(wait_matches) != len(generic_waits):
        raise PatchError("A barrier-wait operation has an unsupported form.")
    ordered_barriers = sorted(
        [(match.start(), "arrive") for match in arrive_matches]
        + [(match.start(), "wait") for match in wait_matches]
    )
    expected_barrier_order = ["arrive", "wait"] * len(arrive_matches)
    if [name for _position, name in ordered_barriers] != expected_barrier_order:
        raise PatchError("PTX barrier arrivals and waits are not ordered in pairs.")
    for arrive, wait in zip(arrive_matches, wait_matches, strict=True):
        if arrive.group(1) != wait.group(3):
            raise PatchError(
                "A barrier arrival and wait use different state registers."
            )
        count = previous_literal_assignment(output, arrive.start(), arrive.group(3))
        if count != 1:
            raise PatchError(f"Unsupported mbarrier arrival count {count}.")

    # The supported 512- and 1024-byte bulk forms become per-lane 16-byte
    # copies. Targets with asynchronous copies wait for their normal copy group;
    # Turing uses synchronous vector loads and stores instead.
    def replace_bulk(match: re.Match[str]) -> str:
        destination = match.group("destination")
        source_address = match.group("source")
        size_register = match.group("size")
        copy_size = previous_literal_assignment(output, match.start(), size_register)
        if copy_size not in (512, 1024):
            raise PatchError(f"Unsupported bulk-copy size {copy_size} bytes.")
        copies = []
        for offset in range(0, copy_size, 512):
            suffix = f"+{offset}" if offset else ""
            if target == TURING:
                copies.extend(
                    [
                        "ld.global.cg.v4.b32 "
                        "{dlss5_copy0, dlss5_copy1, dlss5_copy2, "
                        f"dlss5_copy3}}, [dlss5_source{suffix}];",
                        f"st.shared.v4.b32 [dlss5_address{suffix}], "
                        "{dlss5_copy0, dlss5_copy1, dlss5_copy2, "
                        "dlss5_copy3};",
                    ]
                )
            else:
                copies.append(
                    "cp.async.cg.shared.global "
                    f"[dlss5_address{suffix}], [dlss5_source{suffix}], 16;"
                )
        declarations = ""
        completion = "cp.async.commit_group;\ncp.async.wait_group 0;\n"
        if target == TURING:
            declarations = (
                ".reg .b32 dlss5_copy0, dlss5_copy1, dlss5_copy2, "
                "dlss5_copy3;\n"
            )
            completion = ""
        operations = "\n".join(copies)
        return (
            "{\n"
            ".reg .u32 dlss5_lane, dlss5_offset, dlss5_address;\n"
            ".reg .u64 dlss5_offset64, dlss5_source;\n"
            f"{declarations}"
            "mov.u32 dlss5_lane, %laneid;\n"
            "shl.b32 dlss5_offset, dlss5_lane, 4;\n"
            f"add.s32 dlss5_address, {destination}, dlss5_offset;\n"
            "cvt.u64.u32 dlss5_offset64, dlss5_offset;\n"
            f"add.s64 dlss5_source, {source_address}, dlss5_offset64;\n"
            f"{operations}\n"
            f"{completion}"
            "}"
        )

    output, bulk_count = bulk_pattern.subn(replace_bulk, output)
    stats.add("bulk_copy", bulk_count)

    # The expanded copies run in every lane, so no elected lane is needed.
    output, replaced_elect = elect_pattern.subn(r"mov.pred \1, 1;", output)
    stats.add("elect", replaced_elect)

    # A normal async-copy group does not use the original transaction-size
    # expectation.
    output, expect_count = expect_pattern.subn("", output)
    stats.add("expect_tx", expect_count)

    # Use the pre-Blackwell one-arrival and test-wait barrier forms.
    output, arrive_count = arrive_pattern.subn(
        r"mbarrier.arrive.shared::cta.b64 \1, [\2];", output
    )
    stats.add("barrier_arrive", arrive_count)
    output, wait_count = wait_pattern.subn(
        r"mbarrier.test_wait.shared::cta.b64 \1, [\2], \3;", output
    )
    stats.add("barrier_wait", wait_count)

    # Older targets do not support the four-value vector reduction. Preserve its
    # values as separate packed-half reductions at adjacent addresses.
    reduction_pattern = re.compile(
        rf"(?P<guard>@!?{PTX_PREDICATE}\s+)?"
        r"red\.global\.v4\.f16x2\.add\.noftz\s+"
        r"\[(?P<address>%rd\d+)\],\s*"
        r"\{\s*(?P<value0>%r\d+),\s*(?P<value1>%r\d+),\s*"
        r"(?P<value2>%r\d+),\s*(?P<value3>%r\d+)\s*\}\s*;"
    )
    reduction_matches = list(reduction_pattern.finditer(output))
    if any(match.group("guard") for match in reduction_matches):
        raise PatchError("A predicated vector reduction is unsupported.")

    def replace_reduction(match: re.Match[str]) -> str:
        address = match.group("address")
        values = [match.group(f"value{index}") for index in range(4)]
        lines = []
        for index, value in enumerate(values):
            suffix = f"+{index * 4}" if index else ""
            lines.append(f"red.global.f16x2.add.noftz [{address}{suffix}], {value};")
        return "\n".join(lines)

    output, reduction_count = reduction_pattern.subn(replace_reduction, output)
    stats.add("vector_reduction", reduction_count)

    # Use the acquire-release fence accepted by older targets in place of
    # release-only.
    fence_count = output.count("fence.release.gpu;")
    output = output.replace("fence.release.gpu;", "fence.acq_rel.gpu;")
    stats.add("release_fence", fence_count)

    # Expand fused signed minimum/ReLU into operations older targets support.
    min_relu_pattern = re.compile(
        rf"(?P<guard>@!?{PTX_PREDICATE}\s+)?min\.relu\.s32\s+"
        r"(?P<destination>%r\d+),\s*(?P<first>%r\d+),\s*"
        r"(?P<second>%r\d+)\s*;"
    )
    min_relu_matches = list(min_relu_pattern.finditer(output))
    if any(match.group("guard") for match in min_relu_matches):
        raise PatchError("A predicated fused minimum/ReLU is unsupported.")
    output, min_relu_count = min_relu_pattern.subn(
        r"min.s32 \g<destination>, \g<first>, \g<second>;\n"
        r"max.s32 \g<destination>, \g<destination>, 0;",
        output,
    )
    stats.add("min_relu", min_relu_count)

    if target == TURING:
        output = lower_turing_operations(output, stats)

    unsupported = (
        "elect.sync",
        "cp.async.bulk",
        "mbarrier.expect_tx",
        "mbarrier.try_wait",
        "red.global.v4",
        "min.relu",
        "fence.release.gpu",
    )
    if target.cuda < ADA.cuda:
        unsupported += ("e4m3",)
    if target == TURING:
        unsupported += ("m16n8k16", "min.f16", "max.f16", "cp.async", "mbarrier")
    remaining = [token for token in unsupported if token in output]
    if remaining:
        raise PatchError(
            "Unsupported PTX remains after conversion: " + ", ".join(remaining)
        )
    return output, stats


def is_ngx_architecture(value: int) -> bool:
    return 0x100 <= value <= 0x400 and value & 0xF == 0


def exported_function_range(
    data: bytes | bytearray,
    sections: Sequence[PeSection],
    exports: dict[str, int],
    name: str,
    *,
    maximum_size: int = 4096,
) -> tuple[int, int]:
    start_rva = exports[name]
    start = rva_to_file_offset(start_rva, sections)
    if start is None:
        raise PatchError(f"PE export '{name}' is outside the input file.")
    next_rva = min(
        (rva for rva in set(exports.values()) if rva > start_rva),
        default=start_rva + maximum_size,
    )
    rva_size = min(maximum_size, next_rva - start_rva)
    section_end = len(data)
    for section in sections:
        if section.raw_offset <= start < section.raw_offset + section.raw_size:
            section_end = section.raw_offset + section.raw_size
            break
    end = min(len(data), section_end, start + rva_size)
    if end <= start:
        raise PatchError(f"PE export '{name}' has an invalid code range.")
    return start, end


def decode_stack_immediate(
    data: bytes | bytearray, offset: int, end: int
) -> tuple[int, int, int, int] | None:
    """Decode `mov dword ptr [rsp+displacement], immediate`."""
    if offset + 8 <= end and data[offset : offset + 3] == b"\xc7\x44\x24":
        displacement = struct.unpack_from("<b", data, offset + 3)[0]
        immediate_offset = offset + 4
        value = struct.unpack_from("<I", data, immediate_offset)[0]
        return displacement, immediate_offset, value, 8
    if offset + 11 <= end and data[offset : offset + 3] == b"\xc7\x84\x24":
        displacement = struct.unpack_from("<i", data, offset + 3)[0]
        immediate_offset = offset + 7
        value = struct.unpack_from("<I", data, immediate_offset)[0]
        return displacement, immediate_offset, value, 11
    return None


def find_requirement_architecture(
    data: bytes | bytearray, start: int, end: int, export_name: str
) -> tuple[int, int]:
    assignments: list[tuple[int, int, int, int, int]] = []
    for offset in range(start, end):
        decoded = decode_stack_immediate(data, offset, end)
        if decoded is not None:
            displacement, immediate_offset, value, size = decoded
            assignments.append((offset, displacement, immediate_offset, value, size))

    candidates: list[tuple[int, int]] = []
    for (
        first_offset,
        first_displacement,
        _first_immediate,
        first_value,
        first_size,
    ) in assignments:
        if first_value != 0x12:
            continue
        for (
            second_offset,
            displacement,
            immediate_offset,
            value,
            _second_size,
        ) in assignments:
            if not first_offset + first_size <= second_offset <= first_offset + 64:
                continue
            if displacement != first_displacement + 4:
                continue
            if value == 0 or is_ngx_architecture(value):
                candidates.append((immediate_offset, value))
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise PatchError(
            f"Expected one architecture requirement in export '{export_name}', found "
            f"{len(candidates)}."
        )
    return candidates[0]


def patch_minimum_architectures(
    data: bytearray,
    sections: Sequence[PeSection],
    exports: dict[str, int],
    target_ngx_architecture: int = ADA.ngx,
) -> tuple[ArchitectureRequirementPatch, ...]:
    pattern = re.compile(r"^NVSDK_NGX_([A-Z0-9_]+)_GetFeatureRequirements$")
    requirement_exports = sorted(
        (match.group(1), name)
        for name in exports
        if (match := pattern.fullmatch(name)) is not None
    )
    if not requirement_exports:
        raise PatchError("The PE has no NGX feature-requirement exports.")

    patches: list[ArchitectureRequirementPatch] = []
    for interface, name in requirement_exports:
        start, end = exported_function_range(data, sections, exports, name)
        offset, previous_value = find_requirement_architecture(data, start, end, name)
        patches.append(ArchitectureRequirementPatch(interface, offset, previous_value))

    # Different interface names can alias one implementation. Patch each
    # unique requirement only after every exported implementation validates.
    unique_requirements = dict.fromkeys(
        (patch.offset, patch.previous_value) for patch in patches
    )
    for offset, previous_value in unique_requirements:
        if previous_value > target_ngx_architecture:
            struct.pack_into("<I", data, offset, target_ngx_architecture)
    return tuple(patches)


def patch_exported_architecture(
    data: bytearray,
    sections: Sequence[PeSection],
    exports: dict[str, int],
    target_ngx_architecture: int = ADA.ngx,
) -> tuple[int, int] | None:
    name = "NVSDK_NGX_GetGPUArchitecture"
    if name not in exports:
        return None
    start, end = exported_function_range(data, sections, exports, name, maximum_size=64)
    candidates: list[tuple[int, int]] = []
    for offset in range(start, max(start, end - 5)):
        if data[offset] != 0xB8 or data[offset + 5] != 0xC3:
            continue
        value = struct.unpack_from("<I", data, offset + 1)[0]
        if value == 0 or is_ngx_architecture(value):
            candidates.append((offset + 1, value))
    if len(candidates) != 1:
        raise PatchError(
            f"Expected one architecture value in export '{name}', found "
            f"{len(candidates)}."
        )
    immediate_offset, previous_value = candidates[0]
    if previous_value > target_ngx_architecture:
        struct.pack_into("<I", data, immediate_offset, target_ngx_architecture)
    return immediate_offset, previous_value


def patch_architecture_metadata(
    data: bytearray,
    *,
    lower_architecture: bool,
    target_name: str = ADA.ngx_name,
) -> tuple[int, ...]:
    if not lower_architecture:
        return ()
    key = "NGXGpuArchitecture\0".encode("utf-16le")
    key_offsets: list[int] = []
    position = 0
    while True:
        position = data.find(key, position)
        if position < 0:
            break
        key_offsets.append(position)
        position += len(key)

    patched_offsets: list[int] = []
    for key_offset in key_offsets:
        header_offset = key_offset - 6
        if header_offset < 0:
            raise PatchError("The NGX architecture metadata header is truncated.")
        length, value_length, value_type = struct.unpack_from(
            "<HHH", data, header_offset
        )
        value_offset = (key_offset + len(key) + 3) & ~3
        value_size = value_length * 2
        if (
            value_type != 1
            or not value_length
            or length < value_offset + value_size - header_offset
            or header_offset + length > len(data)
        ):
            raise PatchError("The NGX architecture metadata entry is invalid.")
        try:
            value = bytes(data[value_offset : value_offset + value_size]).decode(
                "utf-16le"
            )
        except UnicodeDecodeError as error:
            raise PatchError(
                "The NGX architecture metadata is not UTF-16LE."
            ) from error
        value = value.split("\0", 1)[0]
        if not value.startswith("NVSDK_NGX_GPU_Arch_"):
            raise PatchError(f"Unexpected NGX architecture metadata value '{value}'.")
        if value == target_name:
            continue
        replacement = (target_name + "\0").encode("utf-16le")
        if len(replacement) > value_size:
            raise PatchError(
                f"The {target_name} value does not fit the metadata entry."
            )
        data[value_offset : value_offset + value_size] = replacement.ljust(
            value_size, b"\0"
        )
        struct.pack_into("<H", data, header_offset + 2, len(target_name) + 1)
        patched_offsets.append(value_offset)
    return tuple(patched_offsets)


def decode_architecture_case_entry(
    data: bytes | bytearray, offset: int, register: int
) -> tuple[int, int] | None:
    if (
        offset < 0
        or offset + 7 > len(data)
        or data[offset] != 0xB8 + register
        or data[offset + 5] != 0xEB
    ):
        return None
    value = struct.unpack_from("<I", data, offset + 1)[0]
    if not is_ngx_architecture(value):
        return None
    target = offset + 7 + struct.unpack_from("<b", data, offset + 6)[0]
    return value, target


def find_success_target(
    data: bytes | bytearray,
    cases_end: int,
    failure_target: int,
    target_ngx_architecture: int = ADA.ngx,
) -> int | None:
    if not cases_end <= failure_target <= len(data):
        return None
    search_end = min(failure_target, cases_end + 64)
    compare = data.find(b"\x3d", cases_end, search_end)
    while compare >= 0:
        if compare + 5 <= len(data):
            lower_bound = struct.unpack_from("<I", data, compare + 1)[0]
            if (
                is_ngx_architecture(lower_bound)
                and lower_bound <= target_ngx_architecture
            ):
                branch = compare + 5
                if branch + 2 == failure_target and data[branch] == 0x7D:
                    target = branch + 2 + struct.unpack_from("<b", data, branch + 1)[0]
                    return target if 0 <= target < len(data) else None
                if (
                    branch + 6 == failure_target
                    and data[branch : branch + 2] == b"\x0f\x8d"
                ):
                    target = branch + 6 + struct.unpack_from("<i", data, branch + 2)[0]
                    return target if 0 <= target < len(data) else None
        compare = data.find(b"\x3d", compare + 1, search_end)
    return None


def find_architecture_cases(
    data: bytes | bytearray,
    target_ngx_architecture: int = ADA.ngx,
) -> list[tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, int, int]] = []
    for register in range(8):
        # RSP cannot hold a normal architecture case value.
        if register == 4:
            continue
        marker = bytes((0xB8 + register,)) + struct.pack(
            "<I", target_ngx_architecture
        ) + b"\xeb"
        position = 0
        while True:
            target_offset = data.find(marker, position)
            if target_offset < 0:
                break
            position = target_offset + 1
            target_entry = decode_architecture_case_entry(
                data, target_offset, register
            )
            if target_entry is None:
                continue
            _target_value, failure_target = target_entry
            start = target_offset
            while True:
                previous = decode_architecture_case_entry(data, start - 7, register)
                if previous is None or previous[1] != failure_target:
                    break
                start -= 7
            end = target_offset + 7
            while True:
                following = decode_architecture_case_entry(data, end, register)
                if following is None or following[1] != failure_target:
                    break
                end += 7
            entries = [
                decode_architecture_case_entry(data, offset, register)
                for offset in range(start, end, 7)
            ]
            values = [entry[0] for entry in entries if entry is not None]
            if (
                len(values) < 4
                or len(values) != len(set(values))
                or min(values) >= target_ngx_architecture
                or max(values) <= target_ngx_architecture
                or not end <= failure_target <= end + 64
            ):
                continue
            success = find_success_target(
                data, end, failure_target, target_ngx_architecture
            )
            if success is not None:
                candidates.append((target_offset, register, failure_target, success))
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        raise PatchError("Cannot find a supported GPU architecture case table.")
    return candidates


def patch_architecture_cases(
    data: bytearray, architectures: Sequence[Architecture]
) -> tuple[ArchitectureCasePatch, ...]:
    pending: list[tuple[Architecture, int, int, int]] = []
    seen_architectures: set[int] = set()
    for architecture in architectures:
        if architecture.ngx in seen_architectures:
            continue
        seen_architectures.add(architecture.ngx)
        try:
            candidates = find_architecture_cases(data, architecture.ngx)
        except PatchError as error:
            raise PatchError(
                f"Cannot find a supported {architecture.name.title()} GPU "
                "architecture case table."
            ) from error
        pending.extend(
            (architecture, offset, register, success)
            for offset, register, _failure, success in candidates
        )

    patches: list[ArchitectureCasePatch] = []
    for architecture, offset, register, success in pending:
        # Redirect the selected entry to the existing success block. Both the
        # short and near forms keep the replacement seven bytes long.
        xor_modrm = 0xC0 | (register << 3) | register
        displacement = success - (offset + 7)
        if -128 <= displacement <= 127:
            replacement = bytes(
                (0x33, xor_modrm, 0x90, 0x90, 0x90, 0xEB, displacement & 0xFF)
            )
        elif -(1 << 31) <= displacement < 1 << 31:
            replacement = bytes((0x33, xor_modrm, 0xE9)) + struct.pack(
                "<i", displacement
            )
        else:
            raise PatchError(
                f"A {architecture.name.title()} success target is out of range."
            )
        data[offset : offset + 7] = replacement
        patches.append(ArchitectureCasePatch(offset, success, architecture.name))
    return tuple(patches)


def patch_ada_cases(data: bytearray) -> tuple[ArchitectureCasePatch, ...]:
    """Patch Ada case tables for callers that use the former helper."""
    return patch_architecture_cases(data, (ADA,))


def patch_ada_case(data: bytearray) -> tuple[int, int]:
    """Patch one case table for callers that use the former helper."""
    patches = patch_ada_cases(data)
    if len(patches) != 1:
        raise PatchError(
            f"Expected one GPU architecture case table, found {len(patches)}."
        )
    return patches[0].offset, patches[0].success_offset


def apply_host_patches(
    data: bytearray, architectures: Sequence[Architecture] = (ADA,)
) -> HostPatchResult:
    if not architectures:
        raise PatchError("At least one target architecture is required.")
    minimum_architecture = min(architectures, key=lambda architecture: architecture.ngx)
    sections = read_pe_sections(data)
    exports = read_pe_exports(data, sections)
    requirements = patch_minimum_architectures(
        data, sections, exports, minimum_architecture.ngx
    )
    architecture_export = patch_exported_architecture(
        data, sections, exports, minimum_architecture.ngx
    )
    previous_architectures = [patch.previous_value for patch in requirements]
    if architecture_export is not None:
        previous_architectures.append(architecture_export[1])
    lower_architecture = any(
        previous > minimum_architecture.ngx for previous in previous_architectures
    )
    metadata_offsets = patch_architecture_metadata(
        data,
        lower_architecture=lower_architecture,
        target_name=minimum_architecture.ngx_name,
    )
    enabled_architectures = tuple(
        architecture
        for architecture in architectures
        if any(previous > architecture.ngx for previous in previous_architectures)
    )
    architecture_cases = (
        patch_architecture_cases(data, enabled_architectures)
        if enabled_architectures
        else ()
    )
    return HostPatchResult(
        requirements,
        architecture_cases,
        architecture_export[0] if architecture_export is not None else None,
        metadata_offsets,
    )


def normalize_architectures(
    architectures: Sequence[Architecture] | Architecture,
) -> tuple[Architecture, ...]:
    if isinstance(architectures, Architecture):
        architectures = (architectures,)
    unique: list[Architecture] = []
    seen_cuda: set[int] = set()
    for architecture in architectures:
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise PatchError(f"Unsupported target architecture '{architecture.name}'.")
        if architecture.cuda not in seen_cuda:
            unique.append(architecture)
            seen_cuda.add(architecture.cuda)
    if not unique:
        raise PatchError("At least one target architecture is required.")
    return tuple(unique)


def repack_fatbin(
    index: int,
    source_blob: bytes,
    tools: CudaTools,
    root: Path,
    architectures: Sequence[Architecture] | Architecture = (ADA,),
) -> tuple[bytes, tuple[ArchitectureBuild, ...]]:
    # Preserve every source image, normalize source PTX line endings, and add
    # every selected cubin that is not already present.
    targets = normalize_architectures(architectures)
    work_dir = root / f"fatbin_{index:02d}"
    work_dir.mkdir(parents=True, exist_ok=False)
    source_path = work_dir / "input.fatbin"
    source_path.write_bytes(source_blob)

    images = extract_images(source_path, work_dir, tools.cuobjdump)
    ptx_images = [image for image in images if image.kind == "ptx"]
    if len(ptx_images) != 1:
        raise PatchError(
            f"Fatbin {index} has {len(ptx_images)} PTX images. Exactly one is required."
        )

    source_ptx_image = ptx_images[0]
    source_ptx_bytes = source_ptx_image.path.read_bytes()
    try:
        source_ptx = source_ptx_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PatchError(f"Fatbin {index} PTX is not UTF-8 text.") from error
    source_arch = parse_ptx_target(source_ptx)
    if source_arch != source_ptx_image.architecture:
        raise PatchError(
            f"Fatbin {index} PTX target sm_{source_arch} does not match its image name."
        )
    if not any(
        image.kind == "elf" and image.architecture == source_arch for image in images
    ):
        raise PatchError(f"Fatbin {index} has no sm_{source_arch} ELF image.")

    elf_images: dict[int, list[Image]] = {}
    for image in images:
        if image.kind == "elf":
            elf_images.setdefault(image.architecture, []).append(image)

    def compile_cubin(
        target: Architecture,
        transformed_path: Path,
        target_cubin: Path,
        *,
        compact: bool,
    ) -> None:
        # Turing's expanded k8 MMA stream is intentionally repetitive. O1 keeps
        # that stream compressible enough for the fixed fatbin allocations.
        ptxas_options = ["--opt-level=1"] if target == TURING or compact else []
        run_tool(
            [
                tools.ptxas,
                f"--gpu-name={target.cuda_name}",
                *ptxas_options,
                transformed_path,
                "--output-file",
                target_cubin,
            ]
        )
        if not target_cubin.is_file() or target_cubin.stat().st_size == 0:
            raise PatchError(
                f"ptxas did not create the {target.cuda_name} cubin for fatbin "
                f"{index}."
            )

    builds: list[ArchitectureBuild] = []
    generated_images: list[Image] = []
    generated_paths: dict[int, tuple[Path, Path]] = {}
    for target in targets:
        existing = elf_images.get(target.cuda, [])
        if len(existing) > 1:
            raise PatchError(
                f"Fatbin {index} contains multiple {target.cuda_name} ELF images."
            )
        if existing:
            builds.append(
                ArchitectureBuild(target, existing[0].path.stat().st_size, False)
            )
            continue

        transformed_ptx, stats = transform_ptx(source_ptx, target)
        transformed_path = work_dir / f"{target.name}.ptx"
        transformed_path.write_bytes(transformed_ptx.encode("utf-8"))
        target_cubin = work_dir / f"{target.name}.{target.cuda_name}.cubin"
        compile_cubin(target, transformed_path, target_cubin, compact=False)
        generated_images.append(Image("elf", target.cuda, target_cubin))
        generated_paths[target.cuda] = (transformed_path, target_cubin)
        builds.append(
            ArchitectureBuild(target, target_cubin.stat().st_size, True, stats)
        )

    deterministic_images: list[Image] = []
    for image_number, image in enumerate(images):
        if image.kind != "ptx":
            deterministic_images.append(image)
            continue
        normalized_path = (
            work_dir / f"source_{image_number}.sm_{image.architecture}.ptx"
        )
        normalized_path.write_bytes(normalize_crlf(image.path.read_bytes()))
        deterministic_images.append(
            Image(image.kind, image.architecture, normalized_path)
        )

    output_path = work_dir / "output.fatbin"

    def build_fatbin() -> bytes:
        output_path.unlink(missing_ok=True)
        command: list[os.PathLike[str] | str] = [
            tools.fatbinary,
            "--64",
            "--compress-all",
            "--compress-mode=size",
            "--create",
            output_path,
        ]
        for image in [*generated_images, *deterministic_images]:
            command.append(
                f"--image3=kind={image.kind},sm={image.architecture},file={image.path}"
            )
        run_tool(command)
        if not output_path.is_file():
            raise PatchError(f"fatbinary did not create fatbin {index}.")
        return preserve_record_flags(output_path.read_bytes(), source_blob)

    generated = build_fatbin()
    # Keep O3 whenever it fits. If a combined build is too large, progressively
    # favor compressibility for the generated pre-Blackwell cubins.
    if len(generated) > len(source_blob):
        for compact_target in (AMPERE, ADA):
            paths = generated_paths.get(compact_target.cuda)
            if paths is None:
                continue
            transformed_path, target_cubin = paths
            compile_cubin(
                compact_target, transformed_path, target_cubin, compact=True
            )
            for build in builds:
                if build.architecture == compact_target:
                    build.cubin_size = target_cubin.stat().st_size
                    break
            generated = build_fatbin()
            if len(generated) <= len(source_blob):
                break

    # Round-trip through cuobjdump and compare every source and generated image.
    verification_path = work_dir / "verified.fatbin"
    verification_path.write_bytes(generated)
    verification_dir = work_dir / "verification"
    verification_dir.mkdir()
    verified_images = extract_images(
        verification_path, verification_dir, tools.cuobjdump
    )

    expected: dict[tuple[str, int], list[bytes]] = {}
    for image in [*images, *generated_images]:
        payload = image.path.read_bytes()
        if image.kind == "ptx":
            payload = canonical_ptx(payload)
        expected.setdefault((image.kind, image.architecture), []).append(payload)

    for image in verified_images:
        key = (image.kind, image.architecture)
        candidates = expected.get(key, [])
        payload = image.path.read_bytes()
        if image.kind == "ptx":
            payload = canonical_ptx(payload)
        try:
            candidates.remove(payload)
        except ValueError as error:
            raise PatchError(
                f"Fatbin {index} changed or added an unexpected {image.kind} sm_"
                f"{image.architecture} image."
            ) from error
    missing = [
        f"{kind} sm_{architecture}"
        for (kind, architecture), payloads in expected.items()
        if payloads
    ]
    if missing:
        raise PatchError(
            f"Fatbin {index} lost these CUDA images: {', '.join(missing)}."
        )
    return generated, tuple(builds)


def path_exists(path: Path) -> bool:
    """Return true for files, directories, and dangling symbolic links."""
    return os.path.lexists(path)


def stage_output(path: Path, data: bytes | bytearray, mode: int) -> Path:
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as target:
            temporary = Path(target.name)
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        temporary.chmod(mode)
        return temporary
    except BaseException as error:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if isinstance(error, OSError):
            raise PatchError(
                f"Cannot stage the output file for '{path}': {error}"
            ) from error
        raise


def atomic_write(
    path: Path,
    data: bytes | bytearray,
    mode: int,
    *,
    replace: bool = True,
) -> None:
    temporary = stage_output(path, data, mode)
    try:
        # Check again after staging because a full CUDA build can take time.
        if not replace and path_exists(path):
            raise PatchError(
                f"Output file now exists: {path}. Use --force to replace it."
            )
        os.replace(temporary, path)
    except OSError as error:
        raise PatchError(f"Cannot write the output file '{path}': {error}") from error
    finally:
        temporary.unlink(missing_ok=True)


def replace_with_backup(
    input_path: Path,
    backup_path: Path,
    data: bytes | bytearray,
    mode: int,
) -> None:
    if path_exists(backup_path):
        raise PatchError(f"Backup file exists: {backup_path}")
    temporary = stage_output(input_path, data, mode)
    moved = False
    try:
        # Check again after the staged DLL has been flushed to disk.
        if path_exists(backup_path):
            raise PatchError(f"Backup file now exists: {backup_path}")
        input_path.rename(backup_path)
        moved = True
        os.replace(temporary, input_path)
    except BaseException as error:
        if moved and path_exists(backup_path) and not path_exists(input_path):
            try:
                backup_path.rename(input_path)
                moved = False
            except OSError as restore_error:
                raise PatchError(
                    f"Cannot install the patched DLL or restore the input. "
                    f"The backup remains at '{backup_path}': {restore_error}"
                ) from error
        if isinstance(error, OSError):
            raise PatchError(f"Cannot install the patched DLL: {error}") from error
        raise
    finally:
        temporary.unlink(missing_ok=True)


def default_backup(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.name}.bak")


def plan_output(
    input_path: Path,
    requested_output: Path | None,
    *,
    force: bool,
    dry_run: bool,
) -> OutputPlan:
    if requested_output is None:
        backup_path = default_backup(input_path)
        if path_exists(backup_path) and not dry_run:
            raise PatchError(f"Backup file exists: {backup_path}")
        return OutputPlan(input_path, backup_path)

    output_path = requested_output.resolve()
    if output_path == input_path:
        raise PatchError("--output cannot name the input DLL.")
    if path_exists(output_path) and not force and not dry_run:
        raise PatchError(
            f"Output file exists: {output_path}. Use --force to replace it."
        )
    return OutputPlan(output_path, None)


def file_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def patch_file(
    input_path: Path,
    output_path: Path | None,
    tools: CudaTools,
    *,
    force: bool,
    dry_run: bool,
    work_dir: Path | None,
    architectures: Sequence[Architecture] | Architecture = SUPPORTED_ARCHITECTURES,
) -> None:
    targets = normalize_architectures(architectures)
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise PatchError(f"Input file does not exist: {input_path}")
    output_plan = plan_output(input_path, output_path, force=force, dry_run=dry_run)

    # Build in memory and delay installation until all fatbins compile and
    # verify. A failure therefore leaves the input DLL in place.
    source_stat = input_path.stat()
    source_bytes = input_path.read_bytes()
    if file_identity(input_path.stat()) != file_identity(source_stat):
        raise PatchError("The input DLL changed while it was being read.")
    data = bytearray(source_bytes)
    authenticode = strip_authenticode(data)
    locations = find_fatbins(data)
    host = apply_host_patches(data, targets)

    print(f"Input:  {input_path}")
    print(f"SHA-256: {sha256_bytes(source_bytes)}")
    print(f"CUDA:   {cuda_version(tools.ptxas)}")
    print("Targets: " + ", ".join(target.cuda_name for target in targets))
    print(f"Fatbins: {len(locations)}")
    if authenticode is None:
        print("Authenticode: no certificate table was present")
    else:
        entry_label = "entry" if authenticode.certificate_count == 1 else "entries"
        print(
            f"Authenticode: removed {authenticode.size} bytes "
            f"({authenticode.certificate_count} {entry_label})"
        )
        if authenticode.trailing_data_size:
            print(
                "Overlay: preserved "
                f"{authenticode.trailing_data_size} bytes after the certificate table"
            )
    requirements = ", ".join(
        f"{patch.interface}@0x{patch.offset:x}" for patch in host.requirements
    )
    architecture_cases = ", ".join(
        f"{patch.architecture}@0x{patch.offset:x}"
        for patch in host.architecture_cases
    )
    print(
        f"Host:    requirements {requirements}; architecture cases "
        f"{architecture_cases or 'none'}"
    )
    host_metadata = []
    if host.architecture_export_offset is not None:
        host_metadata.append(f"export@0x{host.architecture_export_offset:x}")
    host_metadata.extend(
        f"metadata@0x{offset:x}" for offset in host.architecture_metadata_offsets
    )
    if host_metadata:
        print(f"Host:    architecture {', '.join(host_metadata)}")

    temporary_context: tempfile.TemporaryDirectory[str] | None = None
    if work_dir is None:
        label = targets[0].name if len(targets) == 1 else "universal"
        temporary_context = tempfile.TemporaryDirectory(prefix=f"dlssnr-{label}-")
        build_root = Path(temporary_context.name)
    else:
        build_root = work_dir.resolve()
        if build_root.exists():
            if not build_root.is_dir():
                raise PatchError(f"Work path is not a directory: {build_root}")
            if any(build_root.iterdir()):
                raise PatchError(f"Work directory is not empty: {build_root}")
        build_root.mkdir(parents=True, exist_ok=True)

    totals = {target.name: TransformStats() for target in targets}
    try:
        for index, location in enumerate(locations):
            source_blob = bytes(data[location.offset : location.offset + location.size])
            generated, builds = repack_fatbin(
                index, source_blob, tools, build_root, targets
            )
            if len(generated) > location.size:
                raise PatchError(
                    f"Fatbin {index} grew from {location.size} to {len(generated)} bytes. "
                    "The DLL has no space for it."
                )
            # Each rebuilt fatbin must fit its original allocation. Padding
            # the unused tail keeps later PE data at the same file offsets.
            data[location.offset : location.offset + location.size] = generated + bytes(
                location.size - len(generated)
            )
            summaries = []
            for build in builds:
                if build.generated:
                    for name, count in build.stats.values.items():
                        totals[build.architecture.name].add(name, count)
                    detail = f"{build.cubin_size} bytes; {build.stats.summary()}"
                else:
                    detail = f"existing {build.cubin_size} bytes"
                summaries.append(f"{build.architecture.cuda_name} {detail}")
            print(
                f"[{index + 1:02d}/{len(locations):02d}] "
                f"0x{location.offset:x}: {location.size} -> {len(generated)} bytes; "
                + "; ".join(summaries)
            )
    finally:
        if temporary_context is not None:
            temporary_context.cleanup()

    if data == source_bytes:
        raise PatchError("No output bytes changed.")
    # Update the PE checksum after all file changes, including certificate
    # removal.
    pe_checksum = update_pe_checksum(data)
    transform_summaries = [
        f"{target.name}: {totals[target.name].summary()}"
        for target in targets
        if totals[target.name].values
    ]
    print("Transforms: " + ("; ".join(transform_summaries) or "none"))
    print(f"PE checksum: 0x{pe_checksum:08x}")
    print(f"Output SHA-256: {sha256_bytes(data)}")
    if dry_run:
        print("Dry run complete. No output file was written.")
        return

    mode = stat.S_IMODE(source_stat.st_mode)
    if output_plan.backup_path is None:
        atomic_write(output_plan.output_path, data, mode, replace=force)
    else:
        if file_identity(input_path.stat()) != file_identity(source_stat):
            raise PatchError("The input DLL changed during the CUDA build.")
        replace_with_backup(input_path, output_plan.backup_path, data, mode)
        print(f"Backup: {output_plan.backup_path}")
    print(f"Output: {output_plan.output_path}")
    print(
        "Warning: Anti-cheat or file-integrity systems can flag this DLL. "
        "Do not use it with anti-cheat-protected software."
    )


def selected_architectures(args: argparse.Namespace) -> tuple[Architecture, ...]:
    selected = tuple(
        architecture
        for architecture in SUPPORTED_ARCHITECTURES
        if getattr(args, architecture.name)
    )
    return selected or SUPPORTED_ARCHITECTURES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Add selected CUDA architectures and enable their NGX paths in a\n"
            "user-supplied nvngx_dlssnr.dll. All supported RTX generations are\n"
            "selected when no architecture flags are given."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""examples:
  Build for every supported RTX generation (default):
    python %(prog)s "C:\path\to\nvngx_dlssnr.dll"

  Build only the Turing and Ampere targets:
    python %(prog)s --turing --ampere "C:\path\to\nvngx_dlssnr.dll"

  Compile and verify without writing a file:
    python %(prog)s --dry-run "C:\path\to\nvngx_dlssnr.dll"

Architecture flags can be combined. Omit all four to build Turing, Ampere,
Ada, and Blackwell targets together.""",
    )
    parser.add_argument(
        "input", type=Path, help="Path to the original nvngx_dlssnr.dll"
    )
    architecture_group = parser.add_argument_group("target architectures")
    architecture_group.add_argument(
        "-t",
        "--turing",
        action="store_true",
        help="Include Turing / RTX 20 Series (sm_75)",
    )
    architecture_group.add_argument(
        "-A",
        "--ampere",
        action="store_true",
        help="Include Ampere / RTX 30 Series (sm_86)",
    )
    architecture_group.add_argument(
        "-a",
        "--ada",
        action="store_true",
        help="Include Ada / RTX 40 Series (sm_89)",
    )
    architecture_group.add_argument(
        "-b",
        "--blackwell",
        action="store_true",
        help="Include Blackwell / RTX 50 Series (sm_120)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write to this path instead of backing up and replacing the input DLL",
    )
    parser.add_argument(
        "--cuda-bin",
        type=Path,
        help="Directory that contains ptxas, fatbinary, and cuobjdump",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing explicit output file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compile and validate all changes without writing the output",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Keep extracted PTX and generated cubins in this empty directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        parser.print_help(sys.stderr)
        print("\nerror: an input DLL path is required.", file=sys.stderr)
        return 2
    args = parser.parse_args(arguments)
    try:
        tools = find_cuda_tools(args.cuda_bin)
        patch_file(
            args.input,
            args.output,
            tools,
            force=args.force,
            dry_run=args.dry_run,
            work_dir=args.work_dir,
            architectures=selected_architectures(args),
        )
    except PatchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
