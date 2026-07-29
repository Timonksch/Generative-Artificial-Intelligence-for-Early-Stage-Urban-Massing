"""Test public artifact restore helpers without external network access."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import Any

from conftest import REPOSITORY_ROOT
from pytest import MonkeyPatch

DOWNLOAD_ARTIFACTS_SPEC = importlib.util.spec_from_file_location(
    "download_artifacts", REPOSITORY_ROOT / "download_artifacts.py"
)
assert DOWNLOAD_ARTIFACTS_SPEC is not None
assert DOWNLOAD_ARTIFACTS_SPEC.loader is not None
download_artifacts = importlib.util.module_from_spec(DOWNLOAD_ARTIFACTS_SPEC)
sys.modules["download_artifacts"] = download_artifacts
DOWNLOAD_ARTIFACTS_SPEC.loader.exec_module(download_artifacts)


class FakeResponse(io.BytesIO):
    """Small context-manager response used to avoid sockets in tests."""

    def __enter__(self) -> FakeResponse:
        """Return the in-memory response for ``with`` statements."""
        return self

    def __exit__(self, *args: object) -> None:
        """Close the in-memory response after use."""
        self.close()


def test_public_prefix_download_uses_file_index(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Restore one indexed object from a public-prefix manifest."""
    object_payload = b"hello from public r2\n"
    object_sha256 = hashlib.sha256(object_payload).hexdigest()
    index_payload = (
        json.dumps(
            {
                "key": "demo/hello.txt",
                "path": "restored/hello.txt",
                "bytes": len(object_payload),
                "sha256": object_sha256,
            }
        )
        + "\n"
    ).encode()

    manifest = {
        "schema_version": "r2-prefix-manifest-v1",
        "artifacts": [
            {
                "name": "demo",
                "kind": "directory-prefix",
                "local_path": "restored/",
                "r2_prefix": "demo/",
                "s3_uri": "s3://bucket/demo/",
                "public_url": "http://example.invalid/demo/",
                "public_file_index": "file_indexes/demo.jsonl",
            }
        ],
        "restore": {"requirements": [], "commands": []},
    }
    manifest_payload = json.dumps(manifest).encode()

    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    monkeypatch.setattr(download_artifacts, "REPOSITORY_ROOT", repository_root)

    responses = {
        "https://example.test/artifacts_manifest.json": manifest_payload,
        "https://example.test/file_indexes/demo.jsonl": index_payload,
        "https://example.test/demo/hello.txt": object_payload,
    }

    def fake_open_url(url: str, timeout: float) -> Any:
        del timeout
        return FakeResponse(responses[url])

    monkeypatch.setattr(download_artifacts, "_open_url", fake_open_url)
    exit_code = download_artifacts.main(
        [
            "--base-url",
            "https://example.test",
            "--manifest",
            "https://example.test/artifacts_manifest.json",
            "--only",
            "demo",
            "--public-download",
            "--workers",
            "1",
        ]
    )

    assert exit_code == 0
    assert (repository_root / "restored" / "hello.txt").read_text(encoding="utf-8") == (
        "hello from public r2\n"
    )


def test_public_download_uses_manifest_default_artifacts() -> None:
    """Prefer manifest defaults when public download is requested without names."""
    payload = {"restore": {"public_http_default_artifacts": ["all_data", "training_outputs"]}}

    assert download_artifacts._public_download_names(payload, set()) == {
        "all_data",
        "training_outputs",
    }
    assert download_artifacts._public_download_names(payload, {"generated_thesis_dataset"}) == {
        "generated_thesis_dataset"
    }
