import os
import stat
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import dlssnr_ada_patcher as patcher


def make_record(kind: int, flags: int, payload: bytes = b"data") -> bytes:
    header = bytearray(48)
    struct.pack_into("<HHI", header, 0, kind, 0x101, len(header))
    struct.pack_into("<Q", header, 8, len(payload))
    struct.pack_into("<Q", header, 40, flags)
    return bytes(header) + payload


def make_fatbin(*records: bytes) -> bytes:
    payload = b"".join(records)
    return struct.pack("<IHHQ", patcher.FATBIN_MAGIC, 1, 16, len(payload)) + payload


class OutputTests(unittest.TestCase):
    def test_short_output_flag(self) -> None:
        arguments = patcher.build_parser().parse_args(
            ["nvngx_dlssnr.dll", "-o", "patched.dll"]
        )
        self.assertEqual(arguments.output, Path("patched.dll"))

    def test_default_plan_replaces_input_and_uses_backup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "nvngx_dlssnr.dll"
            input_path.write_bytes(b"original")
            plan = patcher.plan_output(
                input_path.resolve(), None, force=False, dry_run=False
            )

            self.assertEqual(plan.output_path, input_path.resolve())
            self.assertEqual(plan.backup_path, Path(f"{input_path.resolve()}.bak"))

    def test_default_plan_rejects_existing_backup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "nvngx_dlssnr.dll"
            input_path.write_bytes(b"original")
            patcher.default_backup(input_path).write_bytes(b"old backup")

            with self.assertRaisesRegex(patcher.PatchError, "Backup file exists"):
                patcher.plan_output(
                    input_path.resolve(), None, force=True, dry_run=False
                )

    def test_explicit_output_cannot_name_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "nvngx_dlssnr.dll"
            input_path.write_bytes(b"original")
            with self.assertRaisesRegex(patcher.PatchError, "cannot name the input"):
                patcher.plan_output(
                    input_path.resolve(), input_path, force=True, dry_run=False
                )

    def test_explicit_output_keeps_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "nvngx_dlssnr.dll"
            output_path = Path(directory) / "patched.dll"
            input_path.write_bytes(b"original")
            plan = patcher.plan_output(
                input_path.resolve(), output_path, force=False, dry_run=False
            )

            patcher.atomic_write(plan.output_path, b"patched", 0o640, replace=False)

            self.assertIsNone(plan.backup_path)
            self.assertEqual(input_path.read_bytes(), b"original")
            self.assertEqual(output_path.read_bytes(), b"patched")

    def test_replace_with_backup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "nvngx_dlssnr.dll"
            backup_path = patcher.default_backup(input_path)
            input_path.write_bytes(b"original")

            patcher.replace_with_backup(input_path, backup_path, b"patched", 0o640)

            self.assertEqual(input_path.read_bytes(), b"patched")
            self.assertEqual(backup_path.read_bytes(), b"original")
            output_mode = stat.S_IMODE(input_path.stat().st_mode)
            if os.name == "nt":
                # Windows chmod only controls the read-only flag.
                self.assertTrue(output_mode & stat.S_IWUSR)
            else:
                self.assertEqual(output_mode, 0o640)

    def test_replace_restores_input_after_install_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "nvngx_dlssnr.dll"
            backup_path = patcher.default_backup(input_path)
            input_path.write_bytes(b"original")

            with (
                mock.patch(
                    "dlssnr_ada_patcher.os.replace", side_effect=OSError("test failure")
                ),
                self.assertRaisesRegex(patcher.PatchError, "Cannot install"),
            ):
                patcher.replace_with_backup(input_path, backup_path, b"patched", 0o640)

            self.assertEqual(input_path.read_bytes(), b"original")
            self.assertFalse(backup_path.exists())

    def test_replace_restores_input_after_keyboard_interrupt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "nvngx_dlssnr.dll"
            backup_path = patcher.default_backup(input_path)
            input_path.write_bytes(b"original")

            with (
                mock.patch(
                    "dlssnr_ada_patcher.os.replace", side_effect=KeyboardInterrupt
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                patcher.replace_with_backup(input_path, backup_path, b"patched", 0o640)

            self.assertEqual(input_path.read_bytes(), b"original")
            self.assertFalse(backup_path.exists())

    def test_explicit_output_rechecks_collision_after_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "patched.dll"
            original_stage = patcher.stage_output

            def stage_then_create(path: Path, data: bytes, mode: int) -> Path:
                temporary = original_stage(path, data, mode)
                output_path.write_bytes(b"other process")
                return temporary

            with (
                mock.patch(
                    "dlssnr_ada_patcher.stage_output", side_effect=stage_then_create
                ),
                self.assertRaisesRegex(patcher.PatchError, "now exists"),
            ):
                patcher.atomic_write(output_path, b"patched", 0o640, replace=False)

            self.assertEqual(output_path.read_bytes(), b"other process")

    def test_tool_launch_error_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tool = Path(directory) / "ptxas"
            tool.write_text("not executable")
            with self.assertRaisesRegex(patcher.PatchError, "Cannot run command"):
                patcher.run_tool([tool, "--version"])


class PeTests(unittest.TestCase):
    @staticmethod
    def make_two_section_pe() -> bytearray:
        data = bytearray(0x800)
        data[:2] = b"MZ"
        struct.pack_into("<I", data, 0x3C, 0x80)
        data[0x80:0x84] = b"PE\0\0"
        struct.pack_into("<HH", data, 0x84, patcher.PE_MACHINE_AMD64, 2)
        struct.pack_into("<H", data, 0x80 + 20, 0xF0)
        optional_offset = 0x80 + 24
        struct.pack_into("<H", data, optional_offset, patcher.PE32_PLUS_MAGIC)
        struct.pack_into("<I", data, optional_offset + 108, 16)
        section_table = optional_offset + 0xF0
        data[section_table : section_table + 8] = b".text\0\0\0"
        struct.pack_into("<IIII", data, section_table + 8, 0x200, 0x1000, 0x200, 0x400)
        section_table += 40
        data[section_table : section_table + 8] = b".rdata\0\0"
        struct.pack_into("<IIII", data, section_table + 8, 0x200, 0x2000, 0x200, 0x600)
        return data

    @staticmethod
    def add_exports(data: bytearray, exports: list[tuple[str, int]]) -> None:
        optional_offset = 0x80 + 24
        export_offset = 0x600
        functions_offset = 0x628
        names_offset = 0x640
        ordinals_offset = 0x660
        strings_offset = 0x670

        def rva(raw_offset: int) -> int:
            return 0x2000 + raw_offset - 0x600

        struct.pack_into("<II", data, optional_offset + 112, rva(export_offset), 0x200)
        struct.pack_into(
            "<IIIII",
            data,
            export_offset + 20,
            len(exports),
            len(exports),
            rva(functions_offset),
            rva(names_offset),
            rva(ordinals_offset),
        )
        cursor = strings_offset
        for index, (name, function_offset) in enumerate(exports):
            function_rva = 0x1000 + function_offset - 0x400
            struct.pack_into("<I", data, functions_offset + index * 4, function_rva)
            struct.pack_into("<I", data, names_offset + index * 4, rva(cursor))
            struct.pack_into("<H", data, ordinals_offset + index * 2, index)
            encoded_name = name.encode("ascii") + b"\0"
            data[cursor : cursor + len(encoded_name)] = encoded_name
            cursor += len(encoded_name)
        if cursor > len(data):
            raise AssertionError("The synthetic export table is too large.")

    def test_update_checksum(self) -> None:
        data = bytearray(512)
        data[:2] = b"MZ"
        struct.pack_into("<I", data, 0x3C, 0x80)
        data[0x80:0x84] = b"PE\0\0"
        struct.pack_into("<HH", data, 0x84, patcher.PE_MACHINE_AMD64, 0)
        struct.pack_into("<H", data, 0x80 + 20, 0xF0)
        struct.pack_into("<H", data, 0x80 + 24, 0x20B)
        data[500:512] = b"checksumtest"

        self.assertEqual(patcher.update_pe_checksum(data), 0xB20E)
        self.assertEqual(struct.unpack_from("<I", data, 0xD8)[0], 0xB20E)
        self.assertEqual(patcher.calculate_pe_checksum(data), 0xB20E)

    def test_strip_authenticode_preserves_trailing_overlay(self) -> None:
        data = self.make_two_section_pe()
        leading_overlay = b"OVERLAY0"
        data.extend(leading_overlay)
        certificate_offset = len(data)

        def certificate(payload: bytes) -> bytes:
            size = 8 + len(payload)
            entry = struct.pack("<IHH", size, 0x200, 2) + payload
            return entry.ljust((size + 7) & ~7, b"\0")

        certificate_table = certificate(b"first") + certificate(b"second entry")
        data.extend(certificate_table)
        trailing_overlay = b"data from a future DLL release"
        data.extend(trailing_overlay)
        original_size = len(data)
        security_directory = 0x80 + 24 + 112 + 4 * 8
        struct.pack_into(
            "<II",
            data,
            security_directory,
            certificate_offset,
            len(certificate_table),
        )

        removal = patcher.strip_authenticode(data)

        if removal is None:
            self.fail("The certificate table was not removed.")
        self.assertEqual(removal.offset, certificate_offset)
        self.assertEqual(removal.size, len(certificate_table))
        self.assertEqual(removal.certificate_count, 2)
        self.assertEqual(removal.trailing_data_size, len(trailing_overlay))
        self.assertEqual(len(data), original_size - len(certificate_table))
        self.assertEqual(data[0x800:certificate_offset], leading_overlay)
        self.assertEqual(data[certificate_offset:], trailing_overlay)
        self.assertEqual(struct.unpack_from("<II", data, security_directory), (0, 0))

    def test_strip_authenticode_accepts_unsigned_pe(self) -> None:
        data = self.make_two_section_pe()
        original = bytes(data)

        self.assertIsNone(patcher.strip_authenticode(data))
        self.assertEqual(data, original)

    def test_strip_authenticode_rejects_partial_security_directory(self) -> None:
        security_directory = 0x80 + 24 + 112 + 4 * 8
        for certificate_offset, certificate_size in ((0, 8), (0x800, 0)):
            with self.subTest(
                offset=certificate_offset,
                size=certificate_size,
            ):
                data = self.make_two_section_pe()
                struct.pack_into(
                    "<II",
                    data,
                    security_directory,
                    certificate_offset,
                    certificate_size,
                )
                original = bytes(data)

                with self.assertRaisesRegex(
                    patcher.PatchError, "security directory is invalid"
                ):
                    patcher.strip_authenticode(data)
                self.assertEqual(data, original)

    def test_strip_authenticode_rejects_misaligned_table(self) -> None:
        data = self.make_two_section_pe()
        data.extend(b"\0")
        certificate_offset = len(data)
        data.extend(struct.pack("<IHH", 8, 0x200, 2))
        security_directory = 0x80 + 24 + 112 + 4 * 8
        struct.pack_into("<II", data, security_directory, certificate_offset, 8)
        original = bytes(data)

        with self.assertRaisesRegex(patcher.PatchError, "not 8-byte aligned"):
            patcher.strip_authenticode(data)
        self.assertEqual(data, original)

    def test_strip_authenticode_rejects_short_entry_length(self) -> None:
        data = self.make_two_section_pe()
        certificate_offset = len(data)
        data.extend(struct.pack("<IHH", 7, 0x200, 2))
        security_directory = 0x80 + 24 + 112 + 4 * 8
        struct.pack_into("<II", data, security_directory, certificate_offset, 8)
        original = bytes(data)

        with self.assertRaisesRegex(patcher.PatchError, "invalid entry length"):
            patcher.strip_authenticode(data)
        self.assertEqual(data, original)

    def test_strip_authenticode_rejects_trailing_entry_header(self) -> None:
        data = self.make_two_section_pe()
        certificate_offset = len(data)
        certificate_table = struct.pack("<IHH", 8, 0x200, 2) + b"next"
        data.extend(certificate_table)
        security_directory = 0x80 + 24 + 112 + 4 * 8
        struct.pack_into(
            "<II",
            data,
            security_directory,
            certificate_offset,
            len(certificate_table),
        )
        original = bytes(data)

        with self.assertRaisesRegex(patcher.PatchError, "truncated entry header"):
            patcher.strip_authenticode(data)
        self.assertEqual(data, original)

    def test_strip_authenticode_rejects_table_in_section_data(self) -> None:
        data = self.make_two_section_pe()
        certificate_offset = 0x700
        struct.pack_into("<IHH", data, certificate_offset, 8, 0x200, 2)
        security_directory = 0x80 + 24 + 112 + 4 * 8
        struct.pack_into("<II", data, security_directory, certificate_offset, 8)

        with self.assertRaisesRegex(patcher.PatchError, "overlaps PE image data"):
            patcher.strip_authenticode(data)

    def test_strip_authenticode_rejects_truncated_entry(self) -> None:
        data = self.make_two_section_pe()
        certificate_offset = len(data)
        data.extend(struct.pack("<IHH", 16, 0x200, 2))
        security_directory = 0x80 + 24 + 112 + 4 * 8
        struct.pack_into("<II", data, security_directory, certificate_offset, 8)
        original = bytes(data)

        with self.assertRaisesRegex(patcher.PatchError, "extends past its table"):
            patcher.strip_authenticode(data)
        self.assertEqual(data, original)

    def test_patch_all_exported_minimum_architectures(self) -> None:
        data = self.make_two_section_pe()
        interface_exports = [
            ("NVSDK_NGX_CUDA_GetFeatureRequirements", 0x410),
            ("NVSDK_NGX_D3D11_GetFeatureRequirements", 0x450),
            ("NVSDK_NGX_D3D12_GetFeatureRequirements", 0x490),
            ("NVSDK_NGX_VULKAN_GetFeatureRequirements", 0x4D0),
            ("NVSDK_NGX_FUTURE_GetFeatureRequirements", 0x510),
        ]
        exports = interface_exports + [("NVSDK_NGX_GetGPUArchitecture", 0x570)]
        self.add_exports(data, exports)
        for _name, function_offset in interface_exports:
            data[function_offset + 8 : function_offset + 24] = (
                b"\xc7\x44\x24\x38\x12\x00\x00\x00\xc7\x44\x24\x3c\xb0\x01\x00\x00"
            )
        data[0x570:0x576] = b"\xb8\xb0\x01\x00\x00\xc3"

        sections = patcher.read_pe_sections(data)
        parsed_exports = patcher.read_pe_exports(data, sections)
        patches = patcher.patch_minimum_architectures(data, sections, parsed_exports)
        architecture_export = patcher.patch_exported_architecture(
            data, sections, parsed_exports
        )

        self.assertEqual(
            [patch.interface for patch in patches],
            ["CUDA", "D3D11", "D3D12", "FUTURE", "VULKAN"],
        )
        for patch in patches:
            self.assertEqual(struct.unpack_from("<I", data, patch.offset)[0], 0x190)
        if architecture_export is None:
            self.fail("The architecture export was not patched.")
        self.assertEqual(
            struct.unpack_from("<I", data, architecture_export[0])[0], 0x190
        )

    def test_patch_architecture_metadata(self) -> None:
        data = bytearray(256)
        header_offset = 16
        key = "NGXGpuArchitecture\0".encode("utf-16le")
        key_offset = header_offset + 6
        value_offset = (key_offset + len(key) + 3) & ~3
        value = "NVSDK_NGX_GPU_Arch_Blackwell2\0".encode("utf-16le")
        length = value_offset + len(value) - header_offset
        struct.pack_into("<HHH", data, header_offset, length, len(value) // 2, 1)
        data[key_offset : key_offset + len(key)] = key
        data[value_offset : value_offset + len(value)] = value

        offsets = patcher.patch_architecture_metadata(data, lower_architecture=True)

        self.assertEqual(offsets, (value_offset,))
        value_length = struct.unpack_from("<H", data, header_offset + 2)[0]
        patched_value = bytes(
            data[value_offset : value_offset + value_length * 2]
        ).decode("utf-16le")
        self.assertEqual(patched_value, patcher.TARGET_NGX_ARCH_NAME + "\0")


class FatbinTests(unittest.TestCase):
    def test_find_and_parse_fatbin(self) -> None:
        fatbin = make_fatbin(make_record(1, 0x8041), make_record(2, 0x1000041))
        data = b"prefix" + fatbin + b"suffix"
        locations = patcher.find_fatbins(data)
        self.assertEqual(len(locations), 1)
        self.assertEqual(locations[0].offset, len(b"prefix"))
        records = patcher.parse_fatbin_records(fatbin)
        self.assertEqual([record.kind for record in records], [1, 2])

    def test_preserve_low_record_flags(self) -> None:
        source = make_fatbin(make_record(1, 0x8041), make_record(2, 0x1000041))
        generated = make_fatbin(
            make_record(2, 0x8011),
            make_record(1, 0x8011),
            make_record(2, 0x1008011),
        )
        output = patcher.preserve_record_flags(generated, source)
        flags = [record.flags for record in patcher.parse_fatbin_records(output)]
        self.assertEqual(flags, [0x8041, 0x8041, 0x1008041])


class PtxTests(unittest.TestCase):
    SOURCE = """
.version 9.4
.target sm_120
.address_size 64
mov.b32 %r2, barrier_storage;
mov.b32 %r3, 512;
{
.reg .pred P_OUT;
elect.sync _|P_OUT, %r9;
selp.b32 %r20, 1, 0, P_OUT;
}
cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%r1], [%rd1], %r3, [%r2];
mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%r2], %r3;
mov.b32 %r4, 1;
mbarrier.arrive.shared::cta.b64 %rd2, [%r2], %r4;
{
.reg .pred P_OUT;
mbarrier.try_wait.shared::cta.b64 P_OUT, [%r2], %rd2;
}
red.global.v4.f16x2.add.noftz [%rd3], {%r5, %r6, %r7, %r8};
fence.release.gpu;
min.relu.s32 %r10, %r11, %r12;
"""

    def test_transform_supported_operations(self) -> None:
        output, stats = patcher.transform_ptx(self.SOURCE)
        self.assertIn(".version 9.3", output)
        self.assertIn(".target sm_89", output)
        self.assertIn("mov.pred P_OUT, 1;", output)
        self.assertEqual(output.count("cp.async.cg.shared.global"), 1)
        self.assertIn("mbarrier.arrive.shared::cta.b64 %rd2, [%r2];", output)
        self.assertIn("mbarrier.test_wait.shared::cta.b64", output)
        self.assertEqual(output.count("red.global.f16x2.add.noftz"), 4)
        self.assertIn("fence.acq_rel.gpu;", output)
        self.assertIn("max.s32 %r10, %r10, 0;", output)
        self.assertEqual(stats.values["bulk_copy"], 1)

    def test_transform_1024_byte_copy(self) -> None:
        source = self.SOURCE.replace("mov.b32 %r3, 512;", "mov.b32 %r3, 1024;")
        output, _stats = patcher.transform_ptx(source)
        self.assertEqual(output.count("cp.async.cg.shared.global"), 2)
        self.assertIn("[dlss5_address+512], [dlss5_source+512]", output)

    def test_reject_mismatched_elect_count(self) -> None:
        source = self.SOURCE.replace("elect.sync _|P_OUT, %r9;", "mov.pred P_OUT, 1;")
        with self.assertRaisesRegex(patcher.PatchError, "ordered in supported groups"):
            patcher.transform_ptx(source)

    def test_reject_reordered_bulk_group(self) -> None:
        bulk = (
            "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
            "[%r1], [%rd1], %r3, [%r2];"
        )
        expectation = "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%r2], %r3;"
        source = self.SOURCE.replace(f"{bulk}\n{expectation}", f"{expectation}\n{bulk}")
        with self.assertRaisesRegex(patcher.PatchError, "ordered in supported groups"):
            patcher.transform_ptx(source)

    def test_reject_mismatched_bulk_expectation(self) -> None:
        source = self.SOURCE.replace(
            "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%r2], %r3;",
            "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%r9], %r3;",
        )
        with self.assertRaisesRegex(patcher.PatchError, "do not match"):
            patcher.transform_ptx(source)

    def test_reject_mismatched_barrier_state(self) -> None:
        source = self.SOURCE.replace(
            "mbarrier.try_wait.shared::cta.b64 P_OUT, [%r2], %rd2;",
            "mbarrier.try_wait.shared::cta.b64 P_OUT, [%r2], %rd9;",
        )
        with self.assertRaisesRegex(patcher.PatchError, "state registers"):
            patcher.transform_ptx(source)

    def test_reject_unknown_bulk_copy_size(self) -> None:
        source = self.SOURCE.replace("mov.b32 %r3, 512;", "mov.b32 %r3, 1536;")
        with self.assertRaisesRegex(patcher.PatchError, "bulk-copy size 1536"):
            patcher.transform_ptx(source)

    def test_canonical_ptx_ignores_text_mode_expansion(self) -> None:
        source = b".version 9.4\r\n.target sm_120\r\n.address_size 64\r\n"
        extracted = source.replace(b"\n", b"\r\n")
        self.assertNotEqual(source, extracted)
        self.assertEqual(
            patcher.canonical_ptx(source), patcher.canonical_ptx(extracted)
        )

    def test_canonical_ptx_keeps_payload_differences(self) -> None:
        self.assertNotEqual(
            patcher.canonical_ptx(b".target sm_120\r\n"),
            patcher.canonical_ptx(b".target sm_89\r\n"),
        )

    def test_canonical_ptx_keeps_carriage_return_inside_token(self) -> None:
        self.assertNotEqual(
            patcher.canonical_ptx(b".target sm_1\r20\n"),
            patcher.canonical_ptx(b".target sm_120\n"),
        )
        self.assertEqual(patcher.canonical_ptx(b"a\rb\n"), b"a\rb\n")


class HostCaseTests(unittest.TestCase):
    @staticmethod
    def add_case_table(
        data: bytearray,
        values: tuple[int, ...],
        *,
        success_distance: int = 20,
        near_branch: bool = False,
    ) -> tuple[int, int, int]:
        register = 6
        start = len(data)
        for value in values:
            data.extend(
                bytes((0xB8 + register,)) + struct.pack("<I", value) + b"\xeb\x00"
            )
        compare = len(data)
        branch_size = 6 if near_branch else 2
        failure = compare + 5 + branch_size
        success = failure + success_distance
        if near_branch:
            data.extend(
                b"\x3d\x40\x01\x00\x00\x0f\x8d" + struct.pack("<i", success - failure)
            )
        else:
            data.extend(
                b"\x3d\x40\x01\x00\x00\x7d" + struct.pack("<b", success - failure)
            )
        data.extend(b"\x90" * success_distance)
        data.extend(b"success")
        for index in range(len(values)):
            entry = start + index * 7
            displacement = failure - (entry + 7)
            if not -128 <= displacement <= 127:
                raise AssertionError("The synthetic case jump is too far.")
            data[entry + 6] = displacement & 0xFF
        ada_case = start + values.index(patcher.TARGET_NGX_ARCH) * 7
        return start, ada_case, success

    def test_patch_ada_case_with_added_future_architecture(self) -> None:
        data = bytearray(b"\x90" * 16)
        values = (0x140, 0x160, 0x170, 0x190, 0x180, 0x1A0, 0x1B0)
        _start, ada_case, success = self.add_case_table(data, values)

        patched_case, target = patcher.patch_ada_case(data)

        self.assertEqual(patched_case, ada_case)
        self.assertEqual(target, success)
        expected = b"\x33\xf6\x90\x90\x90\xeb" + bytes((success - (ada_case + 7),))
        self.assertEqual(data[ada_case : ada_case + 7], expected)

    def test_patch_multiple_ada_case_tables(self) -> None:
        data = bytearray(b"\x90" * 16)
        values = (0x140, 0x160, 0x170, 0x190, 0x180, 0x1A0)
        _first_start, first_ada, first_success = self.add_case_table(data, values)
        data.extend(b"\x90" * 32)
        _second_start, second_ada, second_success = self.add_case_table(data, values)

        patches = patcher.patch_ada_cases(data)

        self.assertEqual(
            [(patch.offset, patch.success_offset) for patch in patches],
            [(first_ada, first_success), (second_ada, second_success)],
        )

    def test_patch_ada_case_with_near_success_branch(self) -> None:
        data = bytearray(b"\x90" * 16)
        values = (0x140, 0x160, 0x170, 0x190, 0x180, 0x1A0)
        _start, ada_case, success = self.add_case_table(
            data, values, success_distance=200, near_branch=True
        )

        patch = patcher.patch_ada_cases(data)[0]

        self.assertEqual((patch.offset, patch.success_offset), (ada_case, success))
        self.assertEqual(data[ada_case : ada_case + 3], b"\x33\xf6\xe9")
        displacement = struct.unpack_from("<i", data, ada_case + 3)[0]
        self.assertEqual(ada_case + 7 + displacement, success)


if __name__ == "__main__":
    unittest.main()
