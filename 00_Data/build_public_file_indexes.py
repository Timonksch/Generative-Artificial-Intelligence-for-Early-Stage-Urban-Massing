"""Build public HTTP file indexes for Cloudflare R2 prefix artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path("artifacts_manifest.json")
DEFAULT_OUTPUT_DIR = Path(".artifacts/file_indexes")
DEFAULT_REGION = "auto"


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON file and require a top-level object."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _artifact_entries(manifest: dict[str, Any], names: set[str]) -> list[dict[str, Any]]:
    """Return selected artifact entries from an R2 prefix manifest."""
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Manifest field 'artifacts' must be a list")

    selected: list[dict[str, Any]] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("Every artifact entry must be an object")
        name = artifact.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Every artifact entry needs a non-empty string 'name'")
        if names and name not in names:
            continue
        selected.append(artifact)

    selected_names = {str(artifact["name"]) for artifact in selected}
    missing_names = sorted(names - selected_names)
    if missing_names:
        raise ValueError(f"Unknown artifact name(s): {', '.join(missing_names)}")
    return selected


def _required_string(artifact: dict[str, Any], field_name: str) -> str:
    """Return a required string field from an artifact entry."""
    value = artifact.get(field_name)
    artifact_name = artifact.get("name", "<unknown>")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Artifact {artifact_name!r} missing field {field_name!r}")
    return value


def _run_aws_json(command: Sequence[str]) -> dict[str, Any]:
    """Run one AWS CLI command and parse a JSON object response."""
    completed = subprocess.run(  # noqa: S603 - executable and arguments are explicit.
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise TypeError("AWS CLI response must be a JSON object")
    return payload


def _list_r2_objects(
    bucket: str,
    prefix: str,
    endpoint_url: str,
    region: str,
    aws_bin: str,
) -> list[dict[str, Any]]:
    """List all objects below an R2 prefix through the S3-compatible API."""
    objects: list[dict[str, Any]] = []
    continuation_token: str | None = None

    while True:
        command = [
            aws_bin,
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            prefix,
            "--endpoint-url",
            endpoint_url,
            "--region",
            region,
            "--output",
            "json",
        ]
        if continuation_token is not None:
            command.extend(["--continuation-token", continuation_token])

        payload = _run_aws_json(command)
        contents = payload.get("Contents", [])
        if not isinstance(contents, list):
            raise TypeError("AWS CLI response field 'Contents' must be a list")
        for item in contents:
            if not isinstance(item, dict):
                raise TypeError("AWS CLI object entry must be an object")
            objects.append(item)

        if not payload.get("IsTruncated"):
            return objects
        raw_token = payload.get("NextContinuationToken")
        if not isinstance(raw_token, str) or not raw_token:
            raise ValueError("Truncated AWS response did not contain NextContinuationToken")
        continuation_token = raw_token


def _index_path_for_artifact(artifact: dict[str, Any], output_dir: Path) -> Path:
    """Return the output JSONL path for one artifact."""
    raw_index = artifact.get("public_file_index")
    if isinstance(raw_index, str) and raw_index:
        return output_dir / Path(raw_index).name
    artifact_name = _required_string(artifact, "name")
    return output_dir / f"{artifact_name}.jsonl"


def _write_index(
    artifact: dict[str, Any],
    objects: Sequence[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Write one newline-delimited JSON public file index."""
    local_path = _required_string(artifact, "local_path").rstrip("/")
    r2_prefix = _required_string(artifact, "r2_prefix").rstrip("/") + "/"
    output_path = _index_path_for_artifact(artifact, output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for item in objects:
        key = item.get("Key")
        size = item.get("Size")
        if not isinstance(key, str) or not key or key.endswith("/"):
            continue
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"Object {key!r} has invalid Size")
        if not key.startswith(r2_prefix):
            raise ValueError(f"Object key {key!r} is outside prefix {r2_prefix!r}")
        relative_key = key[len(r2_prefix) :]
        rows.append({"key": key, "path": f"{local_path}/{relative_key}", "bytes": size})

    rows.sort(key=lambda row: row["path"])
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bucket", help="R2 bucket name. Defaults to manifest bucket.")
    parser.add_argument("--endpoint-url", help="R2 S3 endpoint. Defaults to manifest endpoint.")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--aws-bin", default="aws")
    parser.add_argument("--only", action="append", default=[], help="Artifact name to index.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build public file indexes for all selected prefix artifacts."""
    arguments = _build_parser().parse_args(argv)
    manifest = _load_json_object(arguments.manifest)
    bucket = arguments.bucket or manifest.get("bucket")
    endpoint_url = arguments.endpoint_url or manifest.get("s3_endpoint")
    if not isinstance(bucket, str) or not bucket:
        raise ValueError("Missing R2 bucket; pass --bucket or set manifest 'bucket'")
    if not isinstance(endpoint_url, str) or not endpoint_url:
        raise ValueError("Missing R2 endpoint; pass --endpoint-url or set manifest 's3_endpoint'")

    artifacts = _artifact_entries(manifest, set(arguments.only))
    for artifact in artifacts:
        name = _required_string(artifact, "name")
        prefix = _required_string(artifact, "r2_prefix")
        print(f"[list] {name}: {prefix}")
        objects = _list_r2_objects(
            bucket,
            prefix,
            endpoint_url,
            arguments.region,
            arguments.aws_bin,
        )
        output_path = _write_index(artifact, objects, arguments.output_dir)
        total_bytes = sum(
            int(item["Size"]) for item in objects if isinstance(item.get("Size"), int)
        )
        print(f"[write] {output_path}: {len(objects)} objects, {total_bytes} bytes")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
