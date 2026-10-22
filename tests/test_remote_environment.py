from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SPEC = importlib.util.spec_from_file_location(
    "freeze_remote_environment",
    SCRIPTS / "freeze_remote_environment.py",
)
assert SPEC is not None and SPEC.loader is not None
environment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = environment
SPEC.loader.exec_module(environment)

WRAPPER_SPEC = importlib.util.spec_from_file_location(
    "run_base_sft_in_sealed_environment",
    SCRIPTS / "run_base_sft_in_sealed_environment.py",
)
assert WRAPPER_SPEC is not None and WRAPPER_SPEC.loader is not None
wrapper = importlib.util.module_from_spec(WRAPPER_SPEC)
sys.modules[WRAPPER_SPEC.name] = wrapper
WRAPPER_SPEC.loader.exec_module(wrapper)


def test_runtime_closure_lock_is_explicit_exact_and_separately_bound() -> None:
    assert environment.SCHEMA_VERSION == 4
    assert environment.RUNTIME_CLOSURE_DISTRIBUTION_VERSIONS == {
        "annotated-types": "0.7.0",
        "cffi": "1.17.1",
        "distro": "1.9.0",
        "platformdirs": "4.3.8",
        "pycparser": "2.22",
        "pydantic": "2.11.7",
        "pydantic-core": "2.33.2",
        "typing-inspection": "0.4.1",
    }
    assert len(environment.EXPECTED_DISTRIBUTION_VERSIONS) == 177
    assert len(environment.EXPECTED_VISIBLE_DISTRIBUTION_VERSIONS) == 186
    assert environment.RUNTIME_CLOSURE_LOCK_PATH.read_text(encoding="utf-8").count("==") == 8


def _fake_environment_core(*, include_accelerator: bool) -> dict[str, object]:
    distributions = [
        {
            "name": name,
            "version": environment.EXPECTED_VISIBLE_DISTRIBUTION_VERSIONS[name],
            "file_count": 1,
            "size_bytes": index + 1,
            "tree_sha256": f"{index + 1:064x}",
        }
        for index, name in enumerate(environment.REQUIRED_DISTRIBUTIONS)
    ]
    core: dict[str, object] = {
        "producer": {
            "protocol": environment.PRODUCER_PROTOCOL,
            "implementation": "freeze_remote_environment.py",
            "implementation_sha256": "d" * 64,
            "implementation_size_bytes": 123,
        },
        "python": {
            "implementation": "cpython",
            "cache_tag": "cpython-test",
            "version": "3.test",
            "version_info": [3, 10, 0, "final", 0],
            "abiflags": "",
            "byteorder": "little",
            "executable_path": "/sealed/env/bin/python",
            "resolved_executable_path": "/sealed/env/bin/python3.10",
            "executable_sha256": "e" * 64,
            "executable_size_bytes": 100,
            "executable_mode": 0o755,
            "prefix": "/sealed/env",
            "base_prefix": "/sealed/env",
        },
        "packages": {
            "inventory_scope": environment.PACKAGE_INVENTORY_SCOPE,
            "prefix_tree": {
                "canonical_path": "/sealed/env",
                "file_count": 100,
                "size_bytes": 10_000,
                "tree_sha256": "1" * 64,
                "directories_sha256": "2" * 64,
                "symlink_count": 1,
                "symlinks_sha256": "3" * 64,
            },
            "requirements_lock": {
                "relative_path": environment.REQUIREMENTS_LOCK_RELATIVE_PATH,
                "sha256": environment.REQUIREMENTS_LOCK_SHA256,
                "jiang_source_commit": environment.JIANG_F638_COMMIT,
                "distribution_count": len(environment.EXPECTED_DISTRIBUTION_VERSIONS),
                "versions_sha256": environment.sha256_json(
                    environment.EXPECTED_DISTRIBUTION_VERSIONS
                ),
                "runtime_closure_distributions": dict(
                    environment.RUNTIME_CLOSURE_DISTRIBUTION_VERSIONS
                ),
                "runtime_closure_lock_relative_path": (
                    environment.RUNTIME_CLOSURE_LOCK_RELATIVE_PATH
                ),
                "runtime_closure_lock_sha256": environment.RUNTIME_CLOSURE_LOCK_SHA256,
                "runtime_closure_versions_sha256": environment.sha256_json(
                    environment.RUNTIME_CLOSURE_DISTRIBUTION_VERSIONS
                ),
                "bootstrap_distributions": dict(
                    environment.BOOTSTRAP_DISTRIBUTION_VERSIONS
                ),
                "bootstrap_versions_sha256": environment.sha256_json(
                    environment.BOOTSTRAP_DISTRIBUTION_VERSIONS
                ),
                "visible_distribution_count": len(
                    environment.EXPECTED_VISIBLE_DISTRIBUTION_VERSIONS
                ),
                "visible_versions_sha256": environment.sha256_json(
                    environment.EXPECTED_VISIBLE_DISTRIBUTION_VERSIONS
                ),
            },
            "required_distributions": list(environment.REQUIRED_DISTRIBUTIONS),
            "distributions": distributions,
            "inventory_sha256": environment.sha256_json(distributions),
        },
        "runtime": {
            "platform": {
                "os_name": "posix",
                "sys_platform": "linux",
                "system": "Linux",
                "release": "test",
                "machine": "x86_64",
                "python_build": ["main", "date"],
                "python_compiler": "GCC",
                "libc": ["glibc", "test"],
            },
            "activation": {
                "conda_default_env": "sealed",
                "conda_prefix": "/sealed/env",
                "virtual_env": None,
                "loaded_modules": ["arch/h100", "miniforge/24.9.0"],
                "python_path": "/frozen/source/src:/frozen/source/scripts",
                "sys_path": [
                    "/frozen/source/scripts",
                    "/frozen/source/src",
                    "/sealed/env/lib/python3.10",
                ],
                "path": "/sealed/env/bin:/usr/bin:/bin",
                "ld_library_path": None,
                "ld_preload": None,
            },
            "torch_build": {
                "version": "2.test",
                "cuda": "12.test",
                "hip": None,
                "git_version": "f" * 40,
                "debug": False,
                "cudnn": 90000,
                "config": "PyTorch built with test settings",
            },
        },
    }
    if include_accelerator:
        core["accelerator"] = {
            "expected_gpu_count": 1,
            "torch_devices": [
                {
                    "name": "NVIDIA H100 80GB HBM3",
                    "total_memory_bytes": 80_000 * 1024 * 1024,
                    "compute_capability": [9, 0],
                    "multi_processor_count": 120,
                }
                for _index in range(1)
            ],
            "nvidia_smi": {
                "executable_path": "/usr/bin/nvidia-smi",
                "resolved_executable_path": "/usr/lib/nvidia/nvidia-smi",
                "executable_sha256": "a" * 64,
                "executable_size_bytes": 100,
                "executable_mode": 0o755,
                "devices": [
                    {
                        "driver_version": "test",
                        "name": "NVIDIA H100 80GB HBM3",
                        "memory_total_mib": 80_000,
                    }
                    for _index in range(1)
                ],
            },
        }
    return core


def _write_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, object]]:
    monkeypatch.setattr(environment, "_live_environment_core", _fake_environment_core)
    payload = environment.build_receipt(created_at_utc="2026-08-15T00:00:00Z")
    path = tmp_path / "sealed_remote_environment" / "receipt.json"
    environment._write_new_sealed_receipt_directory(path, payload)
    return path, payload


def test_create_once_receipt_live_reopens_cpu_and_accelerator_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, payload = _write_receipt(tmp_path, monkeypatch)
    assert path.stat().st_mode & 0o222 == 0
    assert payload["receipt_sha256"] == environment.sha256_json(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    cpu_identity = environment.environment_identity(path, require_accelerator=False)
    gpu_identity = environment.environment_identity(path, require_accelerator=True)
    assert cpu_identity == gpu_identity
    assert cpu_identity["receipt_sha256"] == payload["receipt_sha256"]
    assert cpu_identity["identity_sha256"] == environment.sha256_json(
        {key: value for key, value in cpu_identity.items() if key != "identity_sha256"}
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        environment._write_create_once(path, payload)


def test_receipt_rejects_writable_symlink_and_self_hash_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, payload = _write_receipt(tmp_path, monkeypatch)
    path.chmod(0o640)
    with pytest.raises(ValueError, match="read-only"):
        environment.environment_identity(path, require_accelerator=False)
    path.chmod(0o440)

    alias = tmp_path / "environment_alias.json"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="non-symlink"):
        environment.environment_identity(alias, require_accelerator=False)

    path.chmod(0o640)
    tampered = dict(payload)
    tampered["status"] = "forged"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    path.chmod(0o440)
    with pytest.raises(ValueError, match="self-hash mismatch"):
        environment.environment_identity(path, require_accelerator=False)


def test_live_package_and_accelerator_drift_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _payload = _write_receipt(tmp_path, monkeypatch)
    baseline = _fake_environment_core(include_accelerator=True)

    def package_drift(*, include_accelerator: bool) -> dict[str, object]:
        live = copy.deepcopy(baseline)
        live["packages"]["distributions"][0]["version"] = "changed"  # type: ignore[index]
        if not include_accelerator:
            live.pop("accelerator")
        return live

    monkeypatch.setattr(environment, "_live_environment_core", package_drift)
    with pytest.raises(ValueError, match="packages differs"):
        environment.environment_identity(path, require_accelerator=False)

    def accelerator_drift(*, include_accelerator: bool) -> dict[str, object]:
        live = copy.deepcopy(baseline)
        live["accelerator"]["nvidia_smi"]["devices"][0]["driver_version"] = "changed"  # type: ignore[index]
        if not include_accelerator:
            live.pop("accelerator")
        return live

    monkeypatch.setattr(environment, "_live_environment_core", accelerator_drift)
    environment.environment_identity(path, require_accelerator=False)
    with pytest.raises(ValueError, match="accelerator runtime differs"):
        environment.environment_identity(path, require_accelerator=True)


def test_distribution_inventory_hashes_installed_bytes_and_rejects_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "env"
    package = prefix / "lib" / "python" / "site-packages" / "demo"
    metadata = prefix / "lib" / "python" / "site-packages" / "demo-1.dist-info"
    package.mkdir(parents=True)
    metadata.mkdir(parents=True)
    module = package / "__init__.py"
    metadata_file = metadata / "METADATA"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    metadata_file.write_text("Name: demo-pkg\nVersion: 1\n", encoding="utf-8")

    class Distribution:
        version = "1"
        metadata = {"Name": "demo-pkg"}
        files = [
            module.relative_to(prefix),
            metadata_file.relative_to(prefix),
        ]

        @staticmethod
        def locate_file(path: Path) -> Path:
            return prefix / path

    monkeypatch.setattr(
        environment.importlib.metadata,
        "distribution",
        lambda _name: Distribution(),
    )
    first = environment._distribution_inventory("demo-pkg", prefix=prefix)
    module.write_text("VALUE = 2\n", encoding="utf-8")
    second = environment._distribution_inventory("demo-pkg", prefix=prefix)
    assert first["tree_sha256"] != second["tree_sha256"]

    module.unlink()
    module.symlink_to(metadata_file)
    with pytest.raises(ValueError, match="symbolic link"):
        environment._distribution_inventory("demo-pkg", prefix=prefix)


def test_distribution_inventory_accepts_identical_duplicate_metadata_rows_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "env"
    package = prefix / "lib" / "python" / "site-packages" / "demo"
    metadata = prefix / "lib" / "python" / "site-packages" / "demo-1.dist-info"
    package.mkdir(parents=True)
    metadata.mkdir(parents=True)
    module = package / "__init__.py"
    metadata_file = metadata / "METADATA"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    metadata_file.write_text("Name: demo-pkg\nVersion: 1\n", encoding="utf-8")

    class Distribution:
        version = "1"
        metadata = {"Name": "demo-pkg"}
        files = [
            module.relative_to(prefix),
            module.relative_to(prefix),
            metadata_file.relative_to(prefix),
        ]

        @staticmethod
        def locate_file(path: Path) -> Path:
            return prefix / path

    stable_calls: list[Path] = []
    original_stable_file_identity = environment._stable_file_identity

    def traced_stable_file_identity(path: Path) -> tuple[object, ...]:
        stable_calls.append(path)
        return original_stable_file_identity(path)

    monkeypatch.setattr(environment, "_stable_file_identity", traced_stable_file_identity)
    inventory = environment._distribution_inventory(
        "demo-pkg",
        prefix=prefix,
        distribution=Distribution(),
    )
    assert inventory["file_count"] == 2
    assert inventory["size_bytes"] == module.stat().st_size + metadata_file.stat().st_size
    assert len(stable_calls) == 2
    assert set(stable_calls) == {module, metadata_file}


def _synthetic_complete_pip_stale_record_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> list[Path]:
    rows = [Path(environment.PIP_STALE_CONSOLE_RECORD_PATH)]
    rows.extend(
        Path(f"pip/__pycache__/synthetic_{index:03d}.cpython-312.pyc")
        for index in range(environment.PIP_STALE_MISSING_PYC_RECORD_COUNT)
    )
    assert len(rows) == environment.PIP_STALE_MISSING_RECORD_COUNT
    monkeypatch.setattr(
        environment,
        "PIP_STALE_MISSING_PATHS_SHA256",
        environment._missing_metadata_paths_sha256([path.as_posix() for path in rows]),
    )
    return rows


def test_distribution_inventory_accepts_only_the_complete_known_pip_stale_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert environment.PIP_STALE_MISSING_PATHS_SHA256 == (
        "bb087d4ccd6bcafbcf2930356ef97aa0df66039675beb73ae0da8cf52cf0c8f3"
    )
    prefix = tmp_path / "env"
    site_packages = prefix / "lib" / "python3.12" / "site-packages"
    package = site_packages / "pip"
    package.mkdir(parents=True)
    (prefix / "bin").mkdir()
    module = package / "__init__.py"
    module.write_text("__version__ = '25.1.1'\n", encoding="utf-8")
    missing_rows = _synthetic_complete_pip_stale_record_rows(monkeypatch)

    class Distribution:
        version = "25.1.1"
        metadata = {"Name": "pip"}
        files = [module.relative_to(site_packages), *missing_rows]

        @staticmethod
        def locate_file(path: Path) -> Path:
            return site_packages / path

    stable_calls: list[Path] = []
    original_stable_file_identity = environment._stable_file_identity

    def traced_stable_file_identity(path: Path) -> tuple[object, ...]:
        stable_calls.append(path)
        return original_stable_file_identity(path)

    monkeypatch.setattr(environment, "_stable_file_identity", traced_stable_file_identity)
    inventory = environment._distribution_inventory(
        "pip",
        prefix=prefix,
        distribution=Distribution(),
    )
    assert inventory["file_count"] == 1
    assert inventory["size_bytes"] == module.stat().st_size
    assert stable_calls == [module]


def test_complete_pip_stale_set_count_classes_and_digest_are_all_mandatory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _synthetic_complete_pip_stale_record_rows(monkeypatch)
    strings = [path.as_posix() for path in rows]
    environment._validate_pip_stale_missing_metadata_paths(strings)

    with pytest.raises(ValueError, match="stale missing RECORD invariant"):
        environment._validate_pip_stale_missing_metadata_paths(strings[:-1])

    digest_drift = list(strings)
    digest_drift[-1] = "pip/__pycache__/different.cpython-312.pyc"
    with pytest.raises(ValueError, match="stale missing RECORD invariant"):
        environment._validate_pip_stale_missing_metadata_paths(digest_drift)

    wrong_class = list(strings)
    wrong_class[-1] = "pip/__pycache__/different.cpython-311.pyc"
    with pytest.raises(ValueError, match="stale missing RECORD invariant"):
        environment._validate_pip_stale_missing_metadata_paths(wrong_class)


@pytest.mark.parametrize(
    ("metadata_name", "version", "metadata_path"),
    [
        ("pip-extra", "25.1.1", "../../../bin/pip3.12"),
        ("pip", "25.1.2", "../../../bin/pip3.12"),
        ("pip", "25.1.1", "../../../bin/pip3.11"),
    ],
    ids=["wrong-name", "wrong-version", "wrong-record-path"],
)
def test_distribution_inventory_rejects_every_other_missing_record_row(
    tmp_path: Path,
    metadata_name: str,
    version: str,
    metadata_path: str,
) -> None:
    prefix = tmp_path / "env"
    site_packages = prefix / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    (prefix / "bin").mkdir()

    class Distribution:
        metadata = {"Name": metadata_name}
        files = [Path(metadata_path)]

        @staticmethod
        def locate_file(path: Path) -> Path:
            return site_packages / path

    Distribution.version = version
    with pytest.raises(ValueError, match="escapes the active environment"):
        environment._distribution_inventory(
            metadata_name,
            prefix=prefix,
            distribution=Distribution(),
        )


@pytest.mark.parametrize(
    "drift",
    ["appearance", "disappearance", "resolution", "multiplicity"],
)
def test_known_missing_pip_record_row_status_and_multiset_drift_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    prefix = tmp_path / "env"
    site_packages = prefix / "lib" / "python3.12" / "site-packages"
    alternate_site_packages = prefix / "alternate" / "a" / "b"
    site_packages.mkdir(parents=True)
    alternate_site_packages.mkdir(parents=True)
    (prefix / "bin").mkdir()
    (prefix / "alternate" / "bin").mkdir()
    target = prefix / "bin" / "pip3.12"
    missing_rows = _synthetic_complete_pip_stale_record_rows(monkeypatch)
    extra_live_row = Path("pip/__pycache__/extra_live.cpython-312.pyc")
    extra_live_target = site_packages / extra_live_row
    if drift == "disappearance":
        extra_live_target.parent.mkdir(parents=True)
        extra_live_target.write_text("initially present\n", encoding="utf-8")

    class Distribution:
        version = "25.1.1"
        metadata = {"Name": "pip"}
        reads = 0

        @property
        def files(self) -> list[Path]:
            self.reads += 1
            if self.reads == 2:
                if drift == "appearance":
                    target.write_text("appeared\n", encoding="utf-8")
                elif drift == "disappearance":
                    extra_live_target.unlink()
                if drift == "multiplicity":
                    return [*missing_rows, missing_rows[0]]
            rows = list(missing_rows)
            if drift == "disappearance":
                rows.append(extra_live_row)
            return rows

        def locate_file(self, path: Path) -> Path:
            root = (
                alternate_site_packages
                if drift == "resolution" and self.reads == 2
                else site_packages
            )
            return root / path

    expected_error = (
        "metadata path multiset changed"
        if drift == "multiplicity"
        else "file resolution changed"
    )
    with pytest.raises(ValueError, match=expected_error):
        environment._distribution_inventory(
            "pip",
            prefix=prefix,
            distribution=Distribution(),
        )


def test_known_missing_pip_record_row_must_stay_inside_prefix_and_not_be_a_symlink(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "env"
    shallow_site_packages = prefix / "lib" / "site-packages"
    shallow_site_packages.mkdir(parents=True)
    (prefix / "bin").mkdir()
    missing_row = Path("../../../bin/pip3.12")

    class Distribution:
        version = "25.1.1"
        metadata = {"Name": "pip"}
        files = [missing_row]

        @staticmethod
        def locate_file(path: Path) -> Path:
            return shallow_site_packages / path

    with pytest.raises(ValueError, match="lexical normalization"):
        environment._distribution_inventory(
            "pip",
            prefix=prefix,
            distribution=Distribution(),
        )

    site_packages = prefix / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    target = prefix / "bin" / "pip3.12"
    target.symlink_to(prefix / "bin" / "does-not-exist")

    class SymlinkDistribution(Distribution):
        @staticmethod
        def locate_file(path: Path) -> Path:
            return site_packages / path

    with pytest.raises(ValueError, match="symbolic link"):
        environment._distribution_inventory(
            "pip",
            prefix=prefix,
            distribution=SymlinkDistribution(),
        )


@pytest.mark.parametrize(
    "second_rows",
    [
        ("module", "metadata"),
        ("module", "metadata", "metadata"),
    ],
    ids=["multiplicity-drift", "path-drift"],
)
def test_distribution_inventory_rejects_metadata_path_multiset_drift(
    tmp_path: Path,
    second_rows: tuple[str, ...],
) -> None:
    prefix = tmp_path / "env"
    package = prefix / "lib" / "python" / "site-packages" / "demo"
    metadata = prefix / "lib" / "python" / "site-packages" / "demo-1.dist-info"
    package.mkdir(parents=True)
    metadata.mkdir(parents=True)
    module = package / "__init__.py"
    metadata_file = metadata / "METADATA"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    metadata_file.write_text("Name: demo-pkg\nVersion: 1\n", encoding="utf-8")
    rows = {
        "module": module.relative_to(prefix),
        "metadata": metadata_file.relative_to(prefix),
    }

    class Distribution:
        version = "1"
        metadata = {"Name": "demo-pkg"}
        reads = 0

        @property
        def files(self) -> list[Path]:
            self.reads += 1
            names = ("module", "module", "metadata") if self.reads == 1 else second_rows
            return [rows[name] for name in names]

        @staticmethod
        def locate_file(path: Path) -> Path:
            return prefix / path

    with pytest.raises(ValueError, match="metadata path multiset changed"):
        environment._distribution_inventory(
            "demo-pkg",
            prefix=prefix,
            distribution=Distribution(),
        )


def test_full_prefix_tree_is_nonwritable_byte_exact_and_rejects_external_links(
    tmp_path: Path,
) -> None:
    store = tmp_path / "sealed_store"
    prefix = store / "environment"
    library = prefix / "lib"
    library.mkdir(parents=True)
    payload = library / "runtime.so.1"
    payload.write_bytes(b"runtime-v1")
    alias = library / "runtime.so"
    alias.symlink_to(payload.name)
    payload.chmod(0o440)
    library.chmod(0o550)
    prefix.chmod(0o550)
    store.chmod(0o550)

    first = environment._sealed_prefix_tree(prefix)
    assert first["file_count"] == 1
    assert first["symlink_count"] == 1

    store.chmod(0o750)
    prefix.chmod(0o750)
    library.chmod(0o750)
    payload.chmod(0o640)
    payload.write_bytes(b"runtime-v2")
    payload.chmod(0o440)
    library.chmod(0o550)
    prefix.chmod(0o550)
    store.chmod(0o550)
    second = environment._sealed_prefix_tree(prefix)
    assert second["tree_sha256"] != first["tree_sha256"]

    store.chmod(0o750)
    prefix.chmod(0o750)
    library.chmod(0o750)
    alias.unlink()
    external = tmp_path / "external.so"
    external.write_bytes(b"external")
    alias.symlink_to(external)
    library.chmod(0o550)
    prefix.chmod(0o550)
    store.chmod(0o550)
    with pytest.raises(ValueError, match="symlink escapes"):
        environment._sealed_prefix_tree(prefix)
    store.chmod(0o750)
    prefix.chmod(0o750)
    library.chmod(0o750)


def test_full_prefix_tree_accepts_hash_bound_internal_directory_symlink(
    tmp_path: Path,
) -> None:
    store = tmp_path / "sealed_store"
    prefix = store / "environment"
    library = prefix / "lib"
    target = prefix / "share" / "runtime"
    target.mkdir(parents=True)
    library.mkdir()
    payload = target / "runtime.dat"
    payload.write_bytes(b"runtime-v1")
    alias = library / "runtime-current"
    alias.symlink_to(Path("..") / "share" / "runtime", target_is_directory=True)
    payload.chmod(0o440)
    target.chmod(0o550)
    target.parent.chmod(0o550)
    library.chmod(0o550)
    prefix.chmod(0o550)
    store.chmod(0o550)

    first = environment._sealed_prefix_tree(prefix)
    assert first["file_count"] == 1
    assert first["symlink_count"] == 1

    store.chmod(0o750)
    prefix.chmod(0o750)
    target.parent.chmod(0o750)
    target.chmod(0o750)
    payload.chmod(0o640)
    payload.write_bytes(b"runtime-v2")
    payload.chmod(0o440)
    target.chmod(0o550)
    target.parent.chmod(0o550)
    prefix.chmod(0o550)
    store.chmod(0o550)
    second = environment._sealed_prefix_tree(prefix)
    assert second["tree_sha256"] != first["tree_sha256"]


def test_receipt_exact_schema_rejects_rehashed_extra_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(environment, "_live_environment_core", _fake_environment_core)
    payload = environment.build_receipt(created_at_utc="2026-08-15T00:00:00Z")
    payload["untrusted_extra"] = True
    payload["receipt_sha256"] = environment.sha256_json(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="invalid exact schema"):
        environment.validate_receipt(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        "boolean_schema",
        "impossible_timestamp",
        "python_boolean_mode",
        "package_boolean_count",
        "package_extra_field",
        "runtime_missing_config",
        "cross_bound_prefix_mismatch",
        "torch_wrong_compute_capability",
        "nvidia_extra_field",
    ],
)
def test_rehashed_malformed_nested_receipts_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    monkeypatch.setattr(environment, "_live_environment_core", _fake_environment_core)
    payload = environment.build_receipt(created_at_utc="2026-08-15T00:00:00Z")
    if mutation == "boolean_schema":
        payload["schema_version"] = True
    elif mutation == "impossible_timestamp":
        payload["created_at_utc"] = "2026-02-31T00:00:00Z"
    elif mutation == "python_boolean_mode":
        payload["python"]["executable_mode"] = True
    elif mutation == "package_boolean_count":
        payload["packages"]["distributions"][0]["file_count"] = True
        payload["packages"]["inventory_sha256"] = environment.sha256_json(
            payload["packages"]["distributions"]
        )
    elif mutation == "package_extra_field":
        payload["packages"]["prefix_tree"]["untrusted"] = True
    elif mutation == "runtime_missing_config":
        del payload["runtime"]["torch_build"]["config"]
    elif mutation == "cross_bound_prefix_mismatch":
        payload["packages"]["prefix_tree"]["canonical_path"] = "/different/env"
    elif mutation == "torch_wrong_compute_capability":
        payload["accelerator"]["torch_devices"][0]["compute_capability"] = [8, 0]
    elif mutation == "nvidia_extra_field":
        payload["accelerator"]["nvidia_smi"]["devices"][0]["uuid"] = "forged"
    else:  # pragma: no cover - exhaustive parameter list
        raise AssertionError(mutation)
    payload["receipt_sha256"] = environment.sha256_json(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError):
        environment.validate_receipt(payload)


def test_receipt_requires_sealed_parent_and_single_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _payload = _write_receipt(tmp_path, monkeypatch)
    path.parent.chmod(0o750)
    with pytest.raises(ValueError, match="parent.*writable"):
        environment.environment_identity(path, require_accelerator=False)
    path.parent.chmod(0o550)
    alias = tmp_path / "receipt_hardlink.json"
    alias.hardlink_to(path)
    with pytest.raises(ValueError, match="exactly one hard link"):
        environment.environment_identity(path, require_accelerator=False)


def test_package_identity_requires_exact_released_lock_plus_pinned_pip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Distribution:
        def __init__(self, name: str) -> None:
            self.metadata = {"Name": name}

    names = list(environment.REQUIRED_DISTRIBUTIONS)
    distributions = [Distribution(name) for name in names]
    monkeypatch.setattr(
        environment.importlib.metadata,
        "distributions",
        lambda: distributions,
    )
    monkeypatch.setattr(
        environment,
        "_sealed_prefix_tree",
        lambda _prefix: {
            "canonical_path": "/sealed/env",
            "file_count": 1,
            "size_bytes": 1,
            "tree_sha256": "1" * 64,
            "directories_sha256": "2" * 64,
            "symlink_count": 0,
            "symlinks_sha256": "3" * 64,
        },
    )

    def inventory(
        name: str,
        *,
        prefix: Path,
        distribution: object,
    ) -> dict[str, object]:
            del prefix, distribution
            return {
                "name": name,
                "version": environment.EXPECTED_VISIBLE_DISTRIBUTION_VERSIONS[name],
            "file_count": 1,
            "size_bytes": 1,
            "tree_sha256": "4" * 64,
        }

    monkeypatch.setattr(environment, "_distribution_inventory", inventory)
    identity = environment._package_identity(prefix=tmp_path)
    observed = [item["name"] for item in identity["distributions"]]
    assert observed == sorted(names)

    extra = Distribution("transitive-runtime")
    monkeypatch.setattr(
        environment.importlib.metadata,
        "distributions",
        lambda: [*distributions, extra],
    )
    with pytest.raises(ValueError, match="undeclared distributions"):
        environment._package_identity(prefix=tmp_path)

    duplicate = Distribution(environment.REQUIRED_DISTRIBUTIONS[0])
    monkeypatch.setattr(
        environment.importlib.metadata,
        "distributions",
        lambda: [*distributions, duplicate],
    )
    with pytest.raises(ValueError, match="duplicate visible distribution"):
        environment._package_identity(prefix=tmp_path)
    monkeypatch.setattr(
        environment.importlib.metadata,
        "distributions",
        lambda: distributions[:-1],
    )
    with pytest.raises(ValueError, match="required training distributions are missing"):
        environment._package_identity(prefix=tmp_path)


def test_step_wrapper_live_verifies_accelerator_before_exec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    calls: list[dict[str, object]] = []
    executed: list[tuple[str, list[str]]] = []

    def verify(**kwargs: object) -> tuple[dict[str, object], str]:
        calls.append(dict(kwargs))
        return {}, "0" * 64

    def execv(executable: str, argv: list[str]) -> None:
        executed.append((executable, argv))
        raise RuntimeError("exec captured")

    monkeypatch.setattr(wrapper, "verify_campaign_runtime", verify)
    monkeypatch.setattr(wrapper.os, "execv", execv)
    with pytest.raises(RuntimeError, match="exec captured"):
        wrapper.main(
            [
                "--receipt",
                str(receipt),
                "--campaign-contract",
                str(tmp_path / "campaign.json"),
                "--immutable-input-trust-root",
                str(tmp_path / "trust"),
                "--source-root",
                str(tmp_path / "source"),
                "--base-model-path",
                str(tmp_path / "base"),
                "--forbidden-source-registry",
                str(tmp_path / "registry.json"),
                "--reserved-source-registry",
                str(tmp_path / "reserved_registry.json"),
                "--data-root",
                str(tmp_path / "data"),
                "--output-root",
                str(tmp_path / "output"),
                "--task",
                "tsp",
                "--run-contract",
                str(tmp_path / "run_contract.json"),
                "--",
                "-m",
                "torch.distributed.run",
                "--nproc_per_node=4",
            ]
        )
    assert len(calls) == 1
    assert calls[0]["remote_environment_receipt"] == receipt
    assert calls[0]["reserved_source_registry"] == tmp_path / "reserved_registry.json"
    assert calls[0]["require_accelerator"] is True
    assert executed[0][0] == str(Path(sys.executable).resolve())
    assert executed[0][1][1:] == [
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=4",
    ]


def test_publication_docs_keep_real_environment_and_campaign_on_hold() -> None:
    protocol = (ROOT / "experiments" / "sota_20260814" / "PROTOCOL.md").read_text(encoding="utf-8")
    runbook = (ROOT / "docs" / "remote_publication_launch.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for text in (protocol, runbook):
        assert "scripts/freeze_remote_environment.py" in text
        assert "--remote-environment-receipt" in text
        assert "BASE_SFT_REMOTE_ENVIRONMENT_RECEIPT" in text
        assert "BASE_SFT_PYTHON" in text
        assert "HOLD" in text
        assert 'test ! -w "$(dirname "${SEALED_ENV}")"' in text
        assert "--run-contract-authority-root" in text
        assert "--immutable-input-trust-root" in text
        assert "base_sft_resume_fidelity.json" in text
        assert "save_total_limit=64" in text
    assert "No real remote-environment receipt" in runbook
    assert "inside the training Slurm step" in readme
