#!/usr/bin/env python3
"""Verify production wheel contents, metadata, and native payloads."""

from __future__ import annotations

import argparse
import base64
import csv
from email.parser import BytesParser
from email.policy import compat32
import hashlib
import io
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import tempfile
import tomllib
import zipfile


PYTHON_TAGS = ("cp310", "cp311", "cp312", "cp313", "cp314")
ALLOWED_SYSTEM_LIBRARIES = {
    "/System/Library/Frameworks/Foundation.framework/Versions/C/Foundation",
    "/System/Library/Frameworks/Metal.framework/Versions/A/Metal",
    "/usr/lib/libSystem.B.dylib",
    "/usr/lib/libc++.1.dylib",
    "/usr/lib/libobjc.A.dylib",
}
LICENSE_SOURCES = (
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "third_party/metal-cpp/LICENSE.txt",
    "third_party/nanobind/LICENSE.txt",
    "third_party/nanobind/robin_map/LICENSE.txt",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def command(*args: str) -> str:
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
    )
    require(
        result.returncode == 0,
        f"{' '.join(args)} failed:\n{result.stdout}{result.stderr}",
    )
    return result.stdout


def normalized_specifiers(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def validate_zip_paths(archive: zipfile.ZipFile, wheel: Path) -> list[str]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    require(len(names) == len(set(names)), f"{wheel}: duplicate ZIP member")
    for name in names:
        path = PurePosixPath(name)
        require(not path.is_absolute(), f"{wheel}: absolute ZIP path {name}")
        require(".." not in path.parts, f"{wheel}: parent traversal path {name}")
        require("\\" not in name, f"{wheel}: non-POSIX ZIP path {name}")
    return [info.filename for info in infos if not info.is_dir()]


def validate_record(
    archive: zipfile.ZipFile,
    regular_files: list[str],
    record_path: str,
    wheel: Path,
) -> None:
    rows = list(
        csv.reader(io.StringIO(archive.read(record_path).decode("utf-8")))
    )
    require(all(len(row) == 3 for row in rows), f"{wheel}: malformed RECORD")
    record_names = [row[0] for row in rows]
    require(
        len(record_names) == len(set(record_names)),
        f"{wheel}: duplicate RECORD row",
    )
    require(
        set(record_names) == set(regular_files),
        f"{wheel}: RECORD does not cover exactly the regular wheel files",
    )

    for name, digest_field, size_field in rows:
        if name == record_path:
            require(
                not digest_field and not size_field,
                f"{wheel}: RECORD self-row must have empty hash and size",
            )
            continue

        payload = archive.read(name)
        expected_digest = base64.urlsafe_b64encode(
            hashlib.sha256(payload).digest()
        ).rstrip(b"=").decode("ascii")
        require(
            digest_field == f"sha256={expected_digest}",
            f"{wheel}: bad RECORD hash for {name}",
        )
        require(
            size_field == str(len(payload)),
            f"{wheel}: bad RECORD size for {name}",
        )


def split_metadata(metadata_bytes: bytes, wheel: Path) -> tuple[object, bytes]:
    separator = b"\r\n\r\n" if b"\r\n\r\n" in metadata_bytes else b"\n\n"
    headers, found, description = metadata_bytes.partition(separator)
    require(bool(found), f"{wheel}: METADATA has no description separator")
    message = BytesParser(policy=compat32).parsebytes(headers + separator)
    return message, description


def validate_metadata(
    archive: zipfile.ZipFile,
    dist_info: str,
    project: dict,
    project_path: Path,
    wheel: Path,
) -> None:
    metadata_path = f"{dist_info}/METADATA"
    metadata_bytes = archive.read(metadata_path)
    metadata, description = split_metadata(metadata_bytes, wheel)
    project_version = project["version"]

    require(
        metadata.get("Metadata-Version") == "2.4",
        f"{wheel}: unexpected Core Metadata version",
    )
    require(metadata.get("Name") == project["name"], f"{wheel}: wrong name")
    require(
        metadata.get("Version") == project_version,
        f"{wheel}: metadata version does not match pyproject.toml",
    )
    require(
        normalized_specifiers(metadata.get("Requires-Python", ""))
        == normalized_specifiers(project["requires-python"]),
        f"{wheel}: Requires-Python does not match pyproject.toml",
    )
    require(
        "numpy>=1.24" in (metadata.get_all("Requires-Dist") or []),
        f"{wheel}: missing NumPy runtime dependency",
    )
    require(
        metadata.get("License-Expression") == "Apache-2.0",
        f"{wheel}: wrong license expression",
    )
    require(
        metadata.get("Description-Content-Type") == "text/markdown",
        f"{wheel}: README is not declared as Markdown",
    )
    require(
        description == (project_path / "README.md").read_bytes(),
        f"{wheel}: embedded README differs from the source README",
    )

    declared_licenses = set(metadata.get_all("License-File") or [])
    require(
        declared_licenses == set(LICENSE_SOURCES),
        f"{wheel}: incorrect License-File declarations",
    )
    for relative_path in LICENSE_SOURCES:
        packaged_path = f"{dist_info}/licenses/{relative_path}"
        require(
            archive.read(packaged_path)
            == (project_path / relative_path).read_bytes(),
            f"{wheel}: packaged {relative_path} differs from source",
        )


def validate_wheel_metadata(
    archive: zipfile.ZipFile,
    dist_info: str,
    expected_tag: str,
    wheel: Path,
) -> None:
    wheel_path = f"{dist_info}/WHEEL"
    metadata = BytesParser(policy=compat32).parsebytes(archive.read(wheel_path))
    require(
        metadata.get("Wheel-Version") == "1.0",
        f"{wheel}: unsupported wheel metadata version",
    )
    require(
        metadata.get("Root-Is-Purelib") == "false",
        f"{wheel}: native wheel marked as pure Python",
    )
    require(
        metadata.get_all("Tag") == [expected_tag],
        f"{wheel}: internal compatibility tag is not {expected_tag}",
    )
    generators = metadata.get_all("Generator") or []
    require(bool(generators), f"{wheel}: missing wheel generator metadata")
    require(
        all(
            generator.startswith(("scikit-build-core ", "delocate "))
            for generator in generators
        ),
        f"{wheel}: unexpected wheel generator {generators}",
    )


def extract_metallib(extension: bytes, wheel: Path) -> bytes:
    candidates = []
    offset = 0
    while True:
        offset = extension.find(b"MTLB", offset)
        if offset < 0:
            break
        if offset + 24 <= len(extension):
            declared_size = struct.unpack_from("<Q", extension, offset + 16)[0]
            if 96 <= declared_size <= len(extension) - offset:
                candidates.append(extension[offset : offset + declared_size])
        offset += 4
    require(
        len(candidates) == 1,
        f"{wheel}: expected one size-valid embedded MTLB payload, "
        f"found {len(candidates)}",
    )
    return candidates[0]


def expected_metal_kernels(project_path: Path) -> set[str]:
    kernels = set()
    pattern = re.compile(rb"(?m)^kernel\s+void\s+(mg_[A-Za-z0-9_]+)\s*\(")
    for source in (project_path / "src/kernels").glob("*.metal"):
        kernels.update(
            match.decode("ascii") for match in pattern.findall(source.read_bytes())
        )
    require(bool(kernels), "no Metal kernels found in project sources")
    return kernels


def validate_metallib(
    metallib: bytes,
    project_path: Path,
    temporary_path: Path,
    wheel: Path,
) -> str:
    metallib_path = temporary_path / "embedded.metallib"
    metallib_path.write_bytes(metallib)
    symbols = command("xcrun", "metal-nm", str(metallib_path))
    actual_kernels = set(
        re.findall(r"(?m)^[0-9a-fA-F]+\s+T\s+(mg_[A-Za-z0-9_]+)$", symbols)
    )
    expected_kernels = expected_metal_kernels(project_path)
    require(
        actual_kernels == expected_kernels,
        f"{wheel}: embedded metallib kernel set differs from source; "
        f"missing={sorted(expected_kernels - actual_kernels)}, "
        f"extra={sorted(actual_kernels - expected_kernels)}",
    )

    deployment_targets = {
        target.decode("ascii")
        for target in re.findall(
            rb"air64_v[0-9]+-apple-macosx([0-9]+\.[0-9]+\.[0-9]+)",
            metallib,
        )
    }
    require(
        deployment_targets == {"14.0.0"},
        f"{wheel}: Metal AIR targets are {sorted(deployment_targets)}",
    )
    return hashlib.sha256(metallib).hexdigest()


def validate_native_extension(
    extension_path: Path,
    project_path: Path,
    wheel: Path,
) -> str:
    require(
        command("lipo", "-archs", str(extension_path)).strip() == "arm64",
        f"{wheel}: extension is not thin arm64",
    )

    mach_header = command("otool", "-hv", str(extension_path))
    require(
        re.search(
            r"(?m)^MH_MAGIC_64\s+ARM64\s+\S+\s+\S+\s+BUNDLE\s",
            mach_header,
        )
        is not None,
        f"{wheel}: extension is not an arm64 Mach-O bundle",
    )

    linked_libraries = set()
    for line in command("otool", "-L", str(extension_path)).splitlines()[1:]:
        line = line.strip()
        if line:
            linked_libraries.add(line.split(" (", 1)[0])
    require(
        linked_libraries == ALLOWED_SYSTEM_LIBRARIES,
        f"{wheel}: unexpected dynamic libraries: "
        f"{sorted(linked_libraries - ALLOWED_SYSTEM_LIBRARIES)}; "
        f"missing expected libraries: "
        f"{sorted(ALLOWED_SYSTEM_LIBRARIES - linked_libraries)}",
    )

    load_commands = command("otool", "-l", str(extension_path))
    commands = re.findall(r"(?m)^\s*cmd (LC_[A-Z0-9_]+)$", load_commands)
    require(
        commands.count("LC_BUILD_VERSION") == 1,
        f"{wheel}: expected exactly one LC_BUILD_VERSION",
    )
    require("LC_RPATH" not in commands, f"{wheel}: extension contains LC_RPATH")
    require(
        "LC_ID_DYLIB" not in commands,
        f"{wheel}: extension unexpectedly has a dylib install name",
    )
    command_index = next(
        index
        for index, line in enumerate(load_commands.splitlines())
        if line.strip() == "cmd LC_BUILD_VERSION"
    )
    build_version = load_commands.splitlines()[command_index : command_index + 10]
    minimum_os = next(
        line.split()[1]
        for line in build_version
        if line.strip().startswith("minos ")
    )
    require(minimum_os == "14.0", f"{wheel}: binary minos is {minimum_os}")

    command("codesign", "--verify", "--verbose=4", str(extension_path))
    exports = command("nm", "-gU", str(extension_path))
    exported_symbols = {
        line.split()[-1] for line in exports.splitlines() if line.split()
    }
    require(
        "_PyInit__core" in exported_symbols,
        f"{wheel}: missing Python extension initializer",
    )
    require(
        not any(symbol.startswith("_mg_") for symbol in exported_symbols),
        f"{wheel}: standalone C API symbols leaked into the Python extension",
    )

    extension = extension_path.read_bytes()
    metallib = extract_metallib(extension, wheel)
    return validate_metallib(metallib, project_path, extension_path.parent, wheel)


def validate_wheel(
    wheel: Path,
    python_tag: str,
    project: dict,
    project_path: Path,
) -> str:
    project_version = project["version"]
    dist_info = f"metal_graph-{project_version}.dist-info"
    expected_tag = f"{python_tag}-{python_tag}-macosx_14_0_arm64"
    expected_extension = (
        f"metal_graph/_core.cpython-{python_tag.removeprefix('cp')}-darwin.so"
    )

    with zipfile.ZipFile(wheel) as archive:
        regular_files = validate_zip_paths(archive, wheel)
        dist_info_directories = {
            PurePosixPath(name).parts[0]
            for name in regular_files
            if PurePosixPath(name).parts
            and PurePosixPath(name).parts[0].endswith(".dist-info")
        }
        require(
            dist_info_directories == {dist_info},
            f"{wheel}: unexpected dist-info directories "
            f"{sorted(dist_info_directories)}",
        )
        expected_files = {
            f"{dist_info}/METADATA",
            f"{dist_info}/RECORD",
            f"{dist_info}/WHEEL",
            f"{dist_info}/licenses/LICENSE",
            f"{dist_info}/licenses/THIRD_PARTY_NOTICES.md",
            f"{dist_info}/licenses/third_party/metal-cpp/LICENSE.txt",
            f"{dist_info}/licenses/third_party/nanobind/LICENSE.txt",
            (
                f"{dist_info}/licenses/third_party/nanobind/"
                "robin_map/LICENSE.txt"
            ),
            expected_extension,
            "metal_graph/__init__.py",
            "metal_graph/_ids.py",
            "metal_graph/experimental.py",
        }
        actual_files = set(regular_files)
        require(
            actual_files == expected_files,
            f"{wheel}: wheel file set differs from the release allowlist; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}",
        )

        validate_metadata(archive, dist_info, project, project_path, wheel)
        validate_wheel_metadata(archive, dist_info, expected_tag, wheel)
        validate_record(
            archive,
            regular_files,
            f"{dist_info}/RECORD",
            wheel,
        )

        with tempfile.TemporaryDirectory() as directory:
            extension_path = Path(archive.extract(expected_extension, directory))
            return validate_native_extension(extension_path, project_path, wheel)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheelhouse", type=Path)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument(
        "--python-tags",
        nargs="+",
        choices=PYTHON_TAGS,
        default=PYTHON_TAGS,
        help="Expected CPython tags (defaults to the complete release set).",
    )
    args = parser.parse_args()
    project_path = args.project.resolve()

    with (project_path / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    expected_names = {
        tag: (
            f"metal_graph-{project['version']}-{tag}-{tag}"
            "-macosx_14_0_arm64.whl"
        )
        for tag in args.python_tags
    }
    wheels = sorted(args.wheelhouse.glob("*.whl"))
    require(
        {wheel.name for wheel in wheels} == set(expected_names.values()),
        f"wheel set mismatch: {[wheel.name for wheel in wheels]}",
    )

    metallib_hashes = {}
    for python_tag, expected_name in expected_names.items():
        wheel = args.wheelhouse / expected_name
        metallib_hashes[wheel.name] = validate_wheel(
            wheel,
            python_tag,
            project,
            project_path,
        )
    require(
        len(set(metallib_hashes.values())) == 1,
        f"embedded metallibs differ between wheels: {metallib_hashes}",
    )

    metallib_hash = next(iter(metallib_hashes.values()))
    print(
        f"verified {len(wheels)} production wheels; "
        f"embedded metallib sha256={metallib_hash}"
    )


if __name__ == "__main__":
    main()
