"""Inspect and restore external thesis artifacts from a public manifest.

The source repository intentionally excludes large generated datasets, model
checkpoints, and full experiment outputs. This script reads the public
Cloudflare R2 artifact manifest, lists the available artifact prefixes, and
can restore public prefixes when the manifest points to per-file JSONL indexes.
It also prints the S3-compatible restore commands required for full directory
syncs.

Legacy ZIP/TAR archive manifests are also supported for smaller packaged
artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "https://pub-974ec55907ff42e581f011cd95ef519e.r2.dev"
DEFAULT_MANIFEST_NAME = "artifacts_manifest.json"
DEFAULT_DOWNLOAD_DIRECTORY = ".artifacts"
DEFAULT_USER_AGENT = "urban-massing-artifact-client/1.0"
CHUNK_SIZE_BYTES = 1024 * 1024
SHA256_HEX_LENGTH = 64
SUPPORTED_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")
PREFIX_MANIFEST_SCHEMA = "r2-prefix-manifest-v1"
DEFAULT_PUBLIC_DOWNLOAD_WORKERS = 8

REPOSITORY_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Artifact:
    """One external archive declared by a legacy archive manifest."""

    name: str
    file_name: str
    extract_to: Path
    sha256: str | None = None
    size_bytes: int | None = None
    description: str = ""


@dataclass(frozen=True)
class PublicFile:
    """One object declared by a public R2 file index."""

    key: str
    local_path: Path
    size_bytes: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class PrefixRuntimeOptions:
    """Runtime options for R2 prefix-manifest handling."""

    list_only: bool
    public_download: bool
    base_url: str
    timeout: float
    force: bool
    workers: int


def _positive_timeout(value: str) -> float:
    """Parse and validate a positive timeout argument.

    Args:
        value: Command-line timeout value.

    Returns:
        Positive timeout in seconds.

    Raises:
        argparse.ArgumentTypeError: If the value is not positive.

    """
    timeout = float(value)
    if timeout <= 0:
        raise argparse.ArgumentTypeError(f"timeout must be positive, got {value}")
    return timeout


def _resolve_repository_path(raw_path: str) -> Path:
    """Resolve a manifest path and keep it inside the repository.

    Args:
        raw_path: Repository-relative path from the manifest.

    Returns:
        Absolute path below the repository root.

    Raises:
        ValueError: If the path is absolute or escapes the repository.

    """
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise ValueError(f"Manifest path must be repository-relative: {raw_path}")

    repository_root = REPOSITORY_ROOT.absolute()
    resolved = Path(os.path.normpath(repository_root / candidate))
    if not resolved.is_relative_to(repository_root):
        raise ValueError(f"Manifest path escapes repository: {raw_path}")
    return resolved


def _load_json_from_url(url: str, timeout: float) -> dict[str, Any]:
    """Download and parse a JSON object from an HTTP(S) URL.

    Args:
        url: Manifest URL.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If the response is not a JSON object.
        urllib.error.URLError: If the download fails.

    """
    with _open_url(url, timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must contain a JSON object: {url}")
    return payload


def _open_url(url: str, timeout: float) -> Any:
    """Open an HTTP(S) URL with a stable client user agent.

    Args:
        url: HTTP(S) URL to open.
        timeout: Request timeout in seconds.

    Returns:
        A urllib response object.

    Raises:
        ValueError: If the URL scheme is not HTTP(S).
        urllib.error.URLError: If the request fails.

    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Only HTTP(S) URLs are supported: {url}")

    request = urllib.request.Request(  # noqa: S310 - scheme is validated before request creation.
        url, headers={"User-Agent": DEFAULT_USER_AGENT}
    )
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 - scheme is validated above.


def _load_manifest(manifest_location: str, timeout: float) -> dict[str, Any]:
    """Load a manifest from a local path or HTTP(S) URL.

    Args:
        manifest_location: Local manifest path or remote URL.
        timeout: Request timeout for remote manifests.

    Returns:
        Parsed manifest object.

    Raises:
        ValueError: If the manifest is malformed.

    """
    parsed = urllib.parse.urlparse(manifest_location)
    if parsed.scheme in {"http", "https"}:
        return _load_json_from_url(manifest_location, timeout)

    path = Path(manifest_location)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must contain a JSON object: {path}")
    return payload


def _parse_artifact(raw_artifact: object) -> Artifact:
    """Validate one manifest artifact entry.

    Args:
        raw_artifact: Artifact payload from the manifest.

    Returns:
        Validated artifact definition.

    Raises:
        ValueError: If required fields are missing or invalid.

    """
    if not isinstance(raw_artifact, dict):
        raise ValueError("Every artifact entry must be an object")

    name = raw_artifact.get("name")
    file_name = raw_artifact.get("file")
    extract_to = raw_artifact.get("extract_to")
    if not isinstance(name, str) or not name:
        raise ValueError("Artifact field 'name' must be a non-empty string")
    if not isinstance(file_name, str) or not file_name:
        raise ValueError(f"Artifact {name!r} field 'file' must be a non-empty string")
    if not isinstance(extract_to, str) or not extract_to:
        raise ValueError(f"Artifact {name!r} field 'extract_to' must be a non-empty string")

    sha256 = raw_artifact.get("sha256")
    if sha256 is not None and (not isinstance(sha256, str) or len(sha256) != SHA256_HEX_LENGTH):
        raise ValueError(f"Artifact {name!r} field 'sha256' must be a 64-character string")

    size_bytes = raw_artifact.get("bytes")
    if size_bytes is not None and (not isinstance(size_bytes, int) or size_bytes <= 0):
        raise ValueError(f"Artifact {name!r} field 'bytes' must be a positive integer")

    description = raw_artifact.get("description", "")
    if not isinstance(description, str):
        raise ValueError(f"Artifact {name!r} field 'description' must be a string")

    return Artifact(
        name=name,
        file_name=file_name,
        extract_to=_resolve_repository_path(extract_to),
        sha256=sha256,
        size_bytes=size_bytes,
        description=description,
    )


def _parse_manifest(payload: dict[str, Any]) -> list[Artifact]:
    """Validate the manifest and return its artifact entries.

    Args:
        payload: Parsed manifest JSON object.

    Returns:
        List of validated artifact definitions.

    Raises:
        ValueError: If the artifact list is missing or duplicate names exist.

    """
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError("Manifest field 'artifacts' must be a non-empty list")

    artifacts = [_parse_artifact(raw_artifact) for raw_artifact in raw_artifacts]
    names = [artifact.name for artifact in artifacts]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"Duplicate artifact names in manifest: {duplicate_names}")
    return artifacts


def _prefix_artifacts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return artifact entries from an R2 prefix manifest.

    Args:
        payload: Parsed R2 prefix manifest.

    Returns:
        Artifact entries from the manifest.

    Raises:
        ValueError: If the manifest does not contain prefix artifact entries.

    """
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Prefix manifest field 'artifacts' must be a non-empty list")

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("Every prefix artifact entry must be an object")
        for field_name in ("name", "kind", "local_path", "r2_prefix", "s3_uri", "public_url"):
            if not isinstance(artifact.get(field_name), str) or not artifact[field_name]:
                artifact_name = artifact.get("name", "<unknown>")
                raise ValueError(f"Prefix artifact {artifact_name!r} missing field {field_name!r}")
    return artifacts


def _select_prefix_artifacts(
    artifacts: list[dict[str, Any]], requested_names: set[str]
) -> list[dict[str, Any]]:
    """Select R2 prefix artifacts by manifest name.

    Args:
        artifacts: Prefix artifact entries from the manifest.
        requested_names: Requested artifact names.

    Returns:
        Selected prefix artifact entries.

    Raises:
        ValueError: If a requested artifact name is unknown.

    """
    if not requested_names:
        return artifacts

    artifact_by_name = {str(artifact["name"]): artifact for artifact in artifacts}
    unknown_names = sorted(requested_names.difference(artifact_by_name))
    if unknown_names:
        raise ValueError(f"Unknown artifact name(s): {', '.join(unknown_names)}")
    return [artifact_by_name[name] for name in sorted(requested_names)]


def _public_download_names(payload: dict[str, Any], requested_names: set[str]) -> set[str]:
    """Resolve public-download artifact names, honoring manifest defaults."""
    if requested_names:
        return requested_names

    restore = payload.get("restore")
    if not isinstance(restore, dict):
        return requested_names
    default_names = restore.get("public_http_default_artifacts", [])
    if default_names in (None, []):
        return requested_names
    if not isinstance(default_names, list) or not all(
        isinstance(name, str) and name for name in default_names
    ):
        raise ValueError("Prefix manifest field 'restore.public_http_default_artifacts' is invalid")
    return set(default_names)


def _print_prefix_artifacts(artifacts: list[dict[str, Any]]) -> None:
    """Print a compact inventory from an R2 prefix manifest.

    Args:
        artifacts: Selected prefix artifact entries.

    Returns:
        None.

    """
    for artifact in artifacts:
        file_count = artifact.get("files", "unknown")
        size_gib = artifact.get("size_gib", "unknown")
        print(
            f"{artifact['name']}: {artifact['r2_prefix']} -> "
            f"{artifact['local_path']} ({file_count} files, {size_gib} GiB)"
        )


def _print_key_checkpoints(payload: dict[str, Any]) -> None:
    """Print key model checkpoint URLs declared by an R2 prefix manifest.

    Args:
        payload: Parsed R2 prefix manifest.

    Returns:
        None.

    Raises:
        ValueError: If a checkpoint entry is malformed.

    """
    checkpoints = payload.get("key_model_checkpoints", [])
    if not checkpoints:
        return
    if not isinstance(checkpoints, list):
        raise ValueError("Prefix manifest field 'key_model_checkpoints' must be a list")

    print("\nKey model checkpoints:")
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            raise ValueError("Every checkpoint entry must be an object")
        name = checkpoint.get("name")
        local_path = checkpoint.get("local_path")
        public_url = checkpoint.get("public_url")
        if not all(isinstance(value, str) and value for value in (name, local_path, public_url)):
            raise ValueError(f"Malformed checkpoint entry: {checkpoint}")
        print(f"- {name}: {local_path}")
        print(f"  {public_url}")


def _print_restore_commands(payload: dict[str, Any]) -> None:
    """Print S3-compatible restore commands from an R2 prefix manifest.

    Args:
        payload: Parsed R2 prefix manifest.

    Returns:
        None.

    Raises:
        ValueError: If the restore command payload is malformed.

    """
    restore = payload.get("restore")
    if not isinstance(restore, dict):
        raise ValueError("Prefix manifest field 'restore' must be an object")

    requirements = restore.get("requirements", [])
    commands = restore.get("commands", [])
    requirements_are_valid = isinstance(requirements, list) and all(
        isinstance(item, str) for item in requirements
    )
    if not requirements_are_valid:
        raise ValueError("Prefix manifest field 'restore.requirements' must be a string list")
    if not isinstance(commands, list) or not all(isinstance(item, str) for item in commands):
        raise ValueError("Prefix manifest field 'restore.commands' must be a string list")

    print("\nFull directory restore requires S3-compatible access.")
    if requirements:
        print("\nRequirements:")
        for requirement in requirements:
            print(f"- {requirement}")

    print("\nRestore commands:")
    for command in commands:
        print(command)


def _public_file_index_url(artifact: dict[str, Any], base_url: str) -> str | None:
    """Return the public file-index URL declared for a prefix artifact."""
    raw_index = artifact.get("public_file_index")
    if not isinstance(raw_index, str) or not raw_index:
        return None

    parsed = urllib.parse.urlparse(raw_index)
    if parsed.scheme in {"http", "https"}:
        return raw_index
    return _join_url(base_url, raw_index)


def _parse_public_file(raw_file: object) -> PublicFile:
    """Validate one JSONL public file-index entry."""
    if not isinstance(raw_file, dict):
        raise ValueError("Every public file-index entry must be an object")

    key = raw_file.get("key")
    local_path = raw_file.get("path", key)
    if not isinstance(key, str) or not key:
        raise ValueError("Public file-index entry field 'key' must be a non-empty string")
    if not isinstance(local_path, str) or not local_path:
        raise ValueError("Public file-index entry field 'path' must be a non-empty string")

    size_bytes = raw_file.get("bytes")
    if size_bytes is not None and (not isinstance(size_bytes, int) or size_bytes < 0):
        raise ValueError(f"Public file-index entry {key!r} has invalid 'bytes'")

    sha256 = raw_file.get("sha256")
    if sha256 is not None and (not isinstance(sha256, str) or len(sha256) != SHA256_HEX_LENGTH):
        raise ValueError(f"Public file-index entry {key!r} has invalid 'sha256'")

    return PublicFile(
        key=key,
        local_path=_resolve_repository_path(local_path),
        size_bytes=size_bytes,
        sha256=sha256,
    )


def _load_public_file_index(index_url: str, timeout: float) -> list[PublicFile]:
    """Load a newline-delimited JSON file index from a public URL."""
    files: list[PublicFile] = []
    with _open_url(index_url, timeout) as response:
        for line_number, raw_line in enumerate(response, 1):
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            try:
                files.append(_parse_public_file(json.loads(line)))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in public file index {index_url} at line {line_number}"
                ) from error
    if not files:
        raise ValueError(f"Public file index is empty: {index_url}")
    return files


def _existing_public_file_is_valid(path: Path, public_file: PublicFile) -> bool:
    """Return whether an existing file already satisfies index metadata."""
    if not path.is_file():
        return False
    if public_file.size_bytes is not None and path.stat().st_size != public_file.size_bytes:
        return False
    return not (
        public_file.sha256 is not None and _sha256(path).lower() != public_file.sha256.lower()
    )


def _download_public_file(
    public_file: PublicFile,
    base_url: str,
    timeout: float,
    force: bool,
) -> str:
    """Download one indexed public R2 object and verify local metadata."""
    if not force and _existing_public_file_is_valid(public_file.local_path, public_file):
        return "skipped"

    _ensure_directory(public_file.local_path.parent)
    temporary_path = public_file.local_path.with_suffix(public_file.local_path.suffix + ".part")
    object_url = _join_url(base_url, public_file.key)
    try:
        with (
            _open_url(object_url, timeout) as response,
            temporary_path.open("wb") as output,
        ):
            shutil.copyfileobj(response, output, length=CHUNK_SIZE_BYTES)

        if (
            public_file.size_bytes is not None
            and temporary_path.stat().st_size != public_file.size_bytes
        ):
            raise ValueError(
                f"{public_file.key}: size mismatch after download, expected "
                f"{public_file.size_bytes}, got {temporary_path.stat().st_size}"
            )
        if public_file.sha256 is not None:
            actual_sha256 = _sha256(temporary_path)
            if actual_sha256.lower() != public_file.sha256.lower():
                raise ValueError(
                    f"{public_file.key}: SHA-256 mismatch after download, expected "
                    f"{public_file.sha256}, got {actual_sha256}"
                )
        temporary_path.replace(public_file.local_path)
        return "downloaded"
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _download_public_artifact(
    artifact: dict[str, Any],
    base_url: str,
    timeout: float,
    force: bool,
    workers: int,
) -> None:
    """Restore one prefix artifact through public HTTP object URLs."""
    index_url = _public_file_index_url(artifact, base_url)
    if index_url is None:
        raise ValueError(
            f"Prefix artifact {artifact['name']!r} has no 'public_file_index'. "
            "Upload a file-index JSONL object first or use the S3 restore command."
        )

    print(f"[index] {artifact['name']}: {index_url}")
    public_files = _load_public_file_index(index_url, timeout)
    print(f"[restore] {artifact['name']}: {len(public_files)} files")

    counts = {"downloaded": 0, "skipped": 0}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_by_file = {
            executor.submit(
                _download_public_file, public_file, base_url, timeout, force
            ): public_file
            for public_file in public_files
        }
        for completed_count, future in enumerate(as_completed(future_by_file), 1):
            public_file = future_by_file[future]
            try:
                state = future.result()
            except (OSError, ValueError, urllib.error.URLError) as error:
                raise RuntimeError(f"{public_file.key}: {error}") from error
            counts[state] += 1
            if completed_count % 250 == 0 or completed_count == len(public_files):
                print(
                    f"[progress] {artifact['name']}: "
                    f"{completed_count}/{len(public_files)} "
                    f"({counts['downloaded']} downloaded, {counts['skipped']} skipped)"
                )


def _handle_prefix_manifest(
    payload: dict[str, Any],
    requested_names: set[str],
    options: PrefixRuntimeOptions,
) -> int:
    """Handle an R2 prefix manifest.

    Args:
        payload: Parsed R2 prefix manifest.
        requested_names: Requested artifact names.
        options: Runtime behavior for listing, public download, and network access.

    Returns:
        Process exit code.

    """
    effective_names = (
        _public_download_names(payload, requested_names)
        if options.public_download
        else requested_names
    )
    selected_artifacts = _select_prefix_artifacts(_prefix_artifacts(payload), effective_names)
    _print_prefix_artifacts(selected_artifacts)
    _print_key_checkpoints(payload)
    if options.list_only:
        return 0
    if options.public_download:
        for artifact in selected_artifacts:
            _download_public_artifact(
                artifact,
                options.base_url,
                options.timeout,
                options.force,
                options.workers,
            )
        return 0
    _print_restore_commands(payload)
    return 0


def _join_url(base_url: str, file_name: str) -> str:
    """Build an artifact URL from the base URL and manifest file name.

    Args:
        base_url: Public archive base URL.
        file_name: Object name or relative object path.

    Returns:
        Fully qualified artifact URL.

    """
    return f"{base_url.rstrip('/')}/{urllib.parse.quote(file_name.lstrip('/'))}"


def _sha256(path: Path) -> str:
    """Calculate a file SHA-256 digest in bounded chunks.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.

    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, artifact: Artifact) -> None:
    """Verify file size and SHA-256 when manifest values are provided.

    Args:
        path: Downloaded artifact file.
        artifact: Manifest artifact definition.

    Raises:
        ValueError: If size or checksum verification fails.

    """
    if artifact.size_bytes is not None and path.stat().st_size != artifact.size_bytes:
        raise ValueError(
            f"{artifact.name}: size mismatch for {path.name}: "
            f"expected {artifact.size_bytes}, got {path.stat().st_size}"
        )
    if artifact.sha256 is not None:
        actual_sha256 = _sha256(path)
        if actual_sha256.lower() != artifact.sha256.lower():
            raise ValueError(
                f"{artifact.name}: SHA-256 mismatch for {path.name}: "
                f"expected {artifact.sha256}, got {actual_sha256}"
            )


def _download_file(url: str, destination: Path, timeout: float, force: bool) -> None:
    """Download one file through a temporary sibling and atomic replace.

    Args:
        url: Source URL.
        destination: Local destination path.
        timeout: Network timeout in seconds.
        force: Replace an existing local file when true.

    Raises:
        FileExistsError: If the destination exists and force is false.
        urllib.error.URLError: If the request fails.

    """
    if destination.exists() and not force:
        print(f"[skip] {destination} already exists")
        return

    _ensure_directory(destination.parent)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with (
            _open_url(url, timeout) as response,
            temporary_path.open("wb") as output,
        ):
            shutil.copyfileobj(response, output, length=CHUNK_SIZE_BYTES)
        temporary_path.replace(destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _ensure_directory(directory: Path) -> None:
    """Create a repository-local directory when it is missing.

    Args:
        directory: Directory to create, including any missing parents.

    Returns:
        None.

    Raises:
        ValueError: If the directory path is outside the repository.

    """
    resolved_directory = directory.resolve()
    if not resolved_directory.is_relative_to(REPOSITORY_ROOT):
        raise ValueError(f"Directory must be inside repository: {directory}")
    if not resolved_directory.exists():
        relative_directory = resolved_directory.relative_to(REPOSITORY_ROOT)
        print(f"[mkdir] {relative_directory}")
        resolved_directory.mkdir(parents=True, exist_ok=True)


def _archive_members(archive_path: Path) -> list[str]:
    """Return member names from a supported archive without extracting it.

    Args:
        archive_path: Archive file to inspect.

    Returns:
        Archive member names.

    Raises:
        ValueError: If the archive type is not supported.

    """
    archive_name = archive_path.name.lower()
    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            return archive.namelist()
    if archive_name.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(archive_path) as archive:
            return archive.getnames()
    raise ValueError(f"Unsupported archive format: {archive_path.name}")


def _validate_member_path(target_directory: Path, member_name: str) -> None:
    """Ensure one archive member cannot escape the extraction directory.

    Args:
        target_directory: Absolute extraction target.
        member_name: Archive member path.

    Raises:
        ValueError: If the member is absolute or escapes the target directory.

    """
    member_path = Path(member_name)
    if member_path.is_absolute():
        raise ValueError(f"Archive member uses an absolute path: {member_name}")

    resolved_member = (target_directory / member_path).resolve()
    if not resolved_member.is_relative_to(target_directory):
        raise ValueError(f"Archive member escapes extraction target: {member_name}")


def _extract_archive(archive_path: Path, target_directory: Path) -> None:
    """Safely extract a supported archive into a repository directory.

    Args:
        archive_path: Local archive path.
        target_directory: Absolute extraction target below repository root.

    Raises:
        ValueError: If the archive format or any member path is unsafe.

    """
    if not archive_path.name.lower().endswith(SUPPORTED_ARCHIVE_SUFFIXES):
        raise ValueError(f"Unsupported archive format: {archive_path.name}")

    _ensure_directory(target_directory)
    for member_name in _archive_members(archive_path):
        _validate_member_path(target_directory, member_name)

    archive_name = archive_path.name.lower()
    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(target_directory)  # noqa: S202 - all member paths are validated above.
        return

    with tarfile.open(archive_path) as archive:
        archive.extractall(target_directory)  # noqa: S202 - all member paths are validated above.


def _select_artifacts(artifacts: list[Artifact], requested_names: set[str]) -> list[Artifact]:
    """Select requested artifacts and validate all names.

    Args:
        artifacts: All manifest artifacts.
        requested_names: Artifact names passed on the command line.

    Returns:
        Selected artifact definitions.

    Raises:
        ValueError: If an unknown artifact name is requested.

    """
    if not requested_names:
        return artifacts

    artifact_by_name = {artifact.name: artifact for artifact in artifacts}
    unknown_names = sorted(requested_names.difference(artifact_by_name))
    if unknown_names:
        raise ValueError(f"Unknown artifact name(s): {', '.join(unknown_names)}")
    return [artifact_by_name[name] for name in sorted(requested_names)]


def _print_artifacts(artifacts: list[Artifact]) -> None:
    """Print a compact manifest inventory.

    Args:
        artifacts: Artifact definitions to list.

    Returns:
        None.

    """
    for artifact in artifacts:
        checksum_state = "sha256" if artifact.sha256 else "no checksum"
        size_state = f"{artifact.size_bytes:,} bytes" if artifact.size_bytes else "unknown size"
        relative_target = artifact.extract_to.relative_to(REPOSITORY_ROOT)
        print(
            f"{artifact.name}: {artifact.file_name} -> "
            f"{relative_target} ({size_state}, {checksum_state})"
        )


def _default_manifest_url(base_url: str) -> str:
    """Return the default remote manifest URL for a base archive URL.

    Args:
        base_url: Public archive base URL.

    Returns:
        Manifest URL.

    """
    return _join_url(base_url, DEFAULT_MANIFEST_NAME)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Command-line arguments without the executable name.

    Returns:
        Parsed argparse namespace.

    """
    parser = argparse.ArgumentParser(
        description="Inspect or restore external thesis artifacts from a manifest."
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Public archive base URL. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest URL or local JSON path. Defaults to <base-url>/artifacts_manifest.json.",
    )
    parser.add_argument(
        "--download-dir",
        default=DEFAULT_DOWNLOAD_DIRECTORY,
        help=(
            "Local archive cache directory for legacy ZIP/TAR manifests. "
            f"Default: {DEFAULT_DOWNLOAD_DIRECTORY}"
        ),
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Artifact name to select. Repeat to select multiple artifacts. Defaults to all.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List manifest artifacts and exit.",
    )
    parser.add_argument(
        "--public-download",
        action="store_true",
        help=(
            "Download selected prefix artifacts through public HTTP object URLs. "
            "Requires each selected manifest entry to declare 'public_file_index'."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_PUBLIC_DOWNLOAD_WORKERS,
        help=(
            "Parallel public download workers for --public-download. "
            f"Default: {DEFAULT_PUBLIC_DOWNLOAD_WORKERS}."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download archives that already exist for legacy ZIP/TAR manifests.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download and verify without extracting for legacy ZIP/TAR manifests.",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_timeout,
        default=60.0,
        help="Network timeout in seconds. Default: 60.",
    )
    arguments = parser.parse_args(argv)
    if arguments.workers <= 0:
        parser.error("--workers must be positive")
    return arguments


def main(argv: list[str] | None = None) -> int:
    """Run the artifact downloader.

    Args:
        argv: Optional argument list. Uses sys.argv when omitted.

    Returns:
        Process exit code.

    """
    args = parse_args(sys.argv[1:] if argv is None else argv)
    manifest_location = args.manifest or _default_manifest_url(args.base_url)

    try:
        manifest = _load_manifest(manifest_location, args.timeout)
        if manifest.get("schema_version") == PREFIX_MANIFEST_SCHEMA:
            return _handle_prefix_manifest(
                manifest,
                set(args.only),
                PrefixRuntimeOptions(
                    list_only=args.list,
                    public_download=args.public_download,
                    base_url=args.base_url,
                    timeout=args.timeout,
                    force=args.force,
                    workers=args.workers,
                ),
            )

        selected_artifacts = _select_artifacts(_parse_manifest(manifest), set(args.only))
        if args.list:
            _print_artifacts(selected_artifacts)
            return 0

        download_directory = _resolve_repository_path(args.download_dir)
        for artifact in selected_artifacts:
            artifact_url = _join_url(args.base_url, artifact.file_name)
            archive_path = download_directory / Path(artifact.file_name).name
            print(f"[download] {artifact.name}: {artifact_url}")
            _download_file(artifact_url, archive_path, args.timeout, args.force)
            _verify_file(archive_path, artifact)
            if not args.no_extract:
                relative_target = artifact.extract_to.relative_to(REPOSITORY_ROOT)
                print(f"[extract] {archive_path} -> {relative_target}")
                _extract_archive(archive_path, artifact.extract_to)
        return 0
    except (
        OSError,
        RuntimeError,
        ValueError,
        urllib.error.URLError,
        zipfile.BadZipFile,
        tarfile.TarError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
