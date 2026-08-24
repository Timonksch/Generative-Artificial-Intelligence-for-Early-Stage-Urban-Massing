#!/usr/bin/env python3
"""Download and prepare the official Berlin geodata used by the dataset pipeline.

The module downloads the LoD1 CityGML archive and paginated WFS snapshots for
parcels or streets. It writes a manifest with source URLs, timestamps, sizes,
feature counts, and SHA-256 checksums. Network access is bounded by explicit
timeouts and retry limits.

Assumptions:
    - Commands are run from a checkout of this repository.
    - Official Berlin endpoints retain their documented ATOM and WFS interfaces.
    - EPSG:25833 is the common coordinate reference system for local processing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPOSITORY_ROOT / "00_Data" / "01_InputData"
CITYGML_MERGER = Path(__file__).resolve().with_name("01_merge_citygml.py")

LICENSE_NAME = "Data licence Germany - Zero - Version 2.0"
LICENSE_URL = "https://www.govdata.de/dl-de/zero-2-0"
USER_AGENT = "urban-massing-thesis/0.1 (+research data downloader)"

LOD1_ATOM_URL = "https://gdi.berlin.de/data/a_lod1/atom/0.atom"
LOD1_ARCHIVE_URL = "https://gdi.berlin.de/data/a_lod1/atom/LoD1.zip"
PARCELS_CATALOG_URL = "https://daten.berlin.de/datensaetze/alkis-berlin-flurstucke-wfs-1bc014d7"
PARCELS_WFS_URL = "https://gdi.berlin.de/services/wfs/alkis_flurstuecke"
STREETS_WFS_URL = "https://gdi.berlin.de/services/wfs/detailnetz"

DOWNLOAD_CHUNK_BYTES = 1024 * 1024  # One MiB bounds memory during large downloads.
MAX_WFS_PAGE_BYTES = 256 * 1024 * 1024  # Reject unexpectedly large server pages.
MAX_EXTRACTED_BYTES = 20 * 1024 * 1024 * 1024  # Guard against malformed ZIP files.
MAX_RETRY_DELAY_SECONDS = 8.0
MAX_RETRIES = 10
MAX_WFS_PAGE_FEATURES = 50_000


class DownloadError(RuntimeError):
    """Report a failed or invalid official-data download."""


@dataclass(frozen=True)
class WfsSource:
    """Describe one WFS layer and its local output filename."""

    key: str
    title: str
    endpoint: str
    type_name: str
    filename: str


@dataclass(frozen=True)
class DownloadRecord:
    """Capture provenance and integrity metadata for one local artifact."""

    dataset: str
    source_url: str
    local_path: str
    status: str
    size_bytes: int
    sha256: str
    feature_count: int | None = None


@dataclass(frozen=True)
class NetworkOptions:
    """Define bounded settings shared by all network requests."""

    timeout_seconds: float
    retries: int
    page_size: int


WFS_SOURCES = {
    "parcels": WfsSource(
        key="parcels",
        title="ALKIS Berlin parcels",
        endpoint=PARCELS_WFS_URL,
        type_name="alkis_flurstuecke:flurstuecke",
        filename="flurstuecke.geojson",
    ),
    "streets": WfsSource(
        key="streets",
        title="Berlin detailed street network",
        endpoint=STREETS_WFS_URL,
        type_name="detailnetz:c_strassenabschnitte",
        filename="strassenabschnitte.geojson",
    ),
}


def _validate_network_options(timeout_seconds: float, retries: int, page_size: int) -> None:
    """Validate bounded network and WFS pagination settings.

    Args:
        timeout_seconds: Maximum duration of one network operation.
        retries: Maximum number of attempts per request.
        page_size: Maximum number of WFS features requested per page.

    Returns:
        None.

    Raises:
        ValueError: If any setting is outside its safe supported range.

    """
    if timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
    if not 1 <= retries <= MAX_RETRIES:
        raise ValueError(f"retries must be between 1 and {MAX_RETRIES}, got {retries}")
    if not 1 <= page_size <= MAX_WFS_PAGE_FEATURES:
        raise ValueError(
            f"page_size must be between 1 and {MAX_WFS_PAGE_FEATURES}, got {page_size}"
        )


def _retry_delay(attempt: int) -> float:
    """Return a bounded exponential retry delay in seconds."""
    return min(2.0 ** max(attempt - 1, 0), MAX_RETRY_DELAY_SECONDS)


def _https_request(url: str) -> Request:
    """Build a request only for an explicitly validated HTTPS URL.

    Args:
        url: Official source URL that must use HTTPS.

    Returns:
        Request with the project user-agent header.

    Raises:
        ValueError: If the URL does not use the HTTPS scheme.

    """
    if not url.startswith("https://"):
        raise ValueError(f"Only HTTPS source URLs are allowed: {url}")
    return Request(  # noqa: S310 - The scheme is restricted to HTTPS above.
        url,
        headers={"User-Agent": USER_AGENT},
    )


def _request_bytes(url: str, timeout_seconds: float, retries: int) -> bytes:
    """Fetch a bounded in-memory response with retries.

    Args:
        url: Fully encoded HTTPS resource URL.
        timeout_seconds: Per-attempt network timeout.
        retries: Maximum number of attempts.

    Returns:
        Response body as bytes.

    Raises:
        DownloadError: If all attempts fail or the response exceeds the page limit.

    """
    request = _https_request(url)
    last_error: BaseException | None = None

    for attempt in range(1, retries + 1):
        try:
            # The request builder above rejects local and non-HTTPS URL schemes.
            with urlopen(  # noqa: S310 - The request contains a validated HTTPS URL.
                request,
                timeout=timeout_seconds,
            ) as response:
                payload = response.read(MAX_WFS_PAGE_BYTES + 1)
            if len(payload) > MAX_WFS_PAGE_BYTES:
                raise DownloadError(f"Response exceeded {MAX_WFS_PAGE_BYTES} bytes: {url}")
            return payload
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(_retry_delay(attempt))

    raise DownloadError(f"Request failed after {retries} attempts: {url}") from last_error


def _sha256_file(path: Path) -> str:
    """Calculate the SHA-256 checksum of a file without loading it into memory.

    Args:
        path: Existing file whose contents should be hashed.

    Returns:
        Lowercase hexadecimal SHA-256 digest.

    Raises:
        FileNotFoundError: If the input path does not exist.

    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(DOWNLOAD_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_or_absolute(path: Path) -> str:
    """Represent repository files relatively and external paths absolutely."""
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved_path)


def _existing_record(dataset: str, source_url: str, destination: Path) -> DownloadRecord:
    """Build an integrity record for an existing artifact.

    Args:
        dataset: Stable dataset identifier used in the manifest.
        source_url: Official endpoint associated with the artifact.
        destination: Existing local file.

    Returns:
        Manifest record marked as an existing file.

    Raises:
        FileNotFoundError: If destination does not exist.

    """
    return DownloadRecord(
        dataset=dataset,
        source_url=source_url,
        local_path=_relative_or_absolute(destination),
        status="existing",
        size_bytes=destination.stat().st_size,
        sha256=_sha256_file(destination),
    )


def download_binary_file(
    dataset: str,
    source_url: str,
    destination: Path,
    network: NetworkOptions,
    force: bool,
) -> DownloadRecord:
    """Download one large binary file atomically with bounded retries.

    Args:
        dataset: Stable dataset identifier used in the manifest.
        source_url: Official HTTPS download URL.
        destination: Final local file path.
        network: Validated timeout, retry, and page-size settings.
        force: Replace an existing destination when true.

    Returns:
        Provenance and checksum record for the downloaded file.

    Raises:
        DownloadError: If all attempts fail or the response is incomplete.

    """
    if destination.exists() and not force:
        print(f"[skip] {destination} already exists")
        return _existing_record(dataset, source_url, destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    last_error: BaseException | None = None

    for attempt in range(1, network.retries + 1):
        try:
            temporary_path.unlink(missing_ok=True)
            request = _https_request(source_url)
            # The request builder above rejects local and non-HTTPS URL schemes.
            with urlopen(  # noqa: S310 - The request contains a validated HTTPS URL.
                request,
                timeout=network.timeout_seconds,
            ) as response:
                expected_size = int(response.headers.get("Content-Length", "0") or 0)
                with temporary_path.open("wb") as output:
                    while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                        output.write(chunk)

            actual_size = temporary_path.stat().st_size
            if expected_size and actual_size != expected_size:
                raise DownloadError(
                    "Incomplete download for "
                    f"{dataset}: expected {expected_size}, got {actual_size}"
                )
            temporary_path.replace(destination)
            print(f"[ok] {dataset}: {destination} ({actual_size:,} bytes)")
            return DownloadRecord(
                dataset=dataset,
                source_url=source_url,
                local_path=_relative_or_absolute(destination),
                status="downloaded",
                size_bytes=actual_size,
                sha256=_sha256_file(destination),
            )
        except (
            DownloadError,
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            ValueError,
        ) as error:
            last_error = error
            temporary_path.unlink(missing_ok=True)
            if attempt < network.retries:
                time.sleep(_retry_delay(attempt))

    raise DownloadError(
        f"Download failed after {network.retries} attempts: {source_url}"
    ) from last_error


def _build_wfs_url(source: WfsSource, **parameters: str | int) -> str:
    """Build a deterministic WFS request URL for one source layer."""
    base_parameters: dict[str, str | int] = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": source.type_name,
    }
    base_parameters.update(parameters)
    return f"{source.endpoint}?{urlencode(base_parameters)}"


def _wfs_snapshot_url(source: WfsSource) -> str:
    """Return the reproducible unpaged WFS query recorded in the manifest."""
    return _build_wfs_url(
        source,
        outputFormat="application/json",
        srsName="EPSG:25833",
    )


def _fetch_wfs_count(
    source: WfsSource,
    timeout_seconds: float,
    retries: int,
) -> int:
    """Fetch the current feature count for a WFS layer.

    Args:
        source: WFS layer definition.
        timeout_seconds: Per-attempt network timeout.
        retries: Maximum number of request attempts.

    Returns:
        Number of currently matched features.

    Raises:
        DownloadError: If the server response lacks a numeric feature count.

    """
    url = _build_wfs_url(source, resultType="hits")
    payload = _request_bytes(url, timeout_seconds, retries)
    count_match = re.search(rb'\bnumberMatched="([0-9]+)"', payload)
    if count_match is None:
        raise DownloadError(f"Invalid WFS hits response for {source.key}")
    return int(count_match.group(1))


def _fetch_wfs_page(
    source: WfsSource,
    start_index: int,
    page_size: int,
    timeout_seconds: float,
    retries: int,
) -> dict[str, Any]:
    """Fetch and validate one bounded WFS GeoJSON page.

    Args:
        source: WFS layer definition.
        start_index: Zero-based feature offset.
        page_size: Maximum features requested from the server.
        timeout_seconds: Per-attempt network timeout.
        retries: Maximum number of request attempts.

    Returns:
        Parsed GeoJSON FeatureCollection.

    Raises:
        DownloadError: If the response is not a valid FeatureCollection.

    """
    url = _build_wfs_url(
        source,
        outputFormat="application/json",
        srsName="EPSG:25833",
        startIndex=start_index,
        count=page_size,
    )
    payload = _request_bytes(url, timeout_seconds, retries)
    try:
        document = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DownloadError(f"Invalid GeoJSON response for {source.key}") from error

    if not isinstance(document, dict) or document.get("type") != "FeatureCollection":
        raise DownloadError(f"Expected a FeatureCollection for {source.key}")
    if not isinstance(document.get("features"), list):
        raise DownloadError(f"Missing feature list for {source.key}")
    return document


def download_wfs_geojson(
    source: WfsSource,
    destination: Path,
    network: NetworkOptions,
    force: bool,
) -> DownloadRecord:
    """Download a complete WFS layer as one paginated GeoJSON file.

    Args:
        source: WFS layer definition.
        destination: Final local GeoJSON path.
        network: Validated timeout, retry, and page-size settings.
        force: Replace an existing destination when true.

    Returns:
        Provenance record including the downloaded feature count.

    Raises:
        DownloadError: If pagination is incomplete or output writing fails.

    """
    if destination.exists() and not force:
        print(f"[skip] {destination} already exists")
        return _existing_record(source.key, _wfs_snapshot_url(source), destination)

    total_features = _fetch_wfs_count(
        source,
        network.timeout_seconds,
        network.retries,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    written_features = 0
    needs_comma = False

    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            output.write('{"type":"FeatureCollection","crs":')
            json.dump(
                {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::25833"}},
                output,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            output.write(',"features":[')

            for start_index in range(0, total_features, network.page_size):
                page = _fetch_wfs_page(
                    source,
                    start_index,
                    network.page_size,
                    network.timeout_seconds,
                    network.retries,
                )
                features = page["features"]
                if not features:
                    raise DownloadError(
                        f"Pagination stopped at {written_features}/{total_features} features"
                    )
                for feature in features:
                    if needs_comma:
                        output.write(",")
                    json.dump(feature, output, ensure_ascii=False, separators=(",", ":"))
                    needs_comma = True
                    written_features += 1
                print(f"[page] {source.key}: {written_features:,}/{total_features:,}")

            output.write("]}")

        if written_features != total_features:
            raise DownloadError(
                f"Feature count mismatch for {source.key}: {written_features}/{total_features}"
            )
        temporary_path.replace(destination)
    except (DownloadError, OSError, TypeError, ValueError) as error:
        temporary_path.unlink(missing_ok=True)
        raise DownloadError(f"Failed to write WFS snapshot for {source.key}") from error

    size_bytes = destination.stat().st_size
    print(f"[ok] {source.title}: {destination} ({written_features:,} features)")
    return DownloadRecord(
        dataset=source.key,
        source_url=_wfs_snapshot_url(source),
        local_path=_relative_or_absolute(destination),
        status="downloaded",
        size_bytes=size_bytes,
        sha256=_sha256_file(destination),
        feature_count=written_features,
    )


def _safe_zip_target(destination: Path, member_name: str) -> Path:
    """Resolve a ZIP member and reject paths escaping the extraction root.

    Args:
        destination: Intended extraction directory.
        member_name: Archive member path supplied by the ZIP file.

    Returns:
        Safe resolved destination path for the member.

    Raises:
        DownloadError: If the member would escape the destination directory.

    """
    destination_root = destination.resolve()
    target = (destination_root / member_name).resolve()
    if target != destination_root and destination_root not in target.parents:
        raise DownloadError(f"Unsafe ZIP member path: {member_name}")
    return target


def extract_lod1_archive(archive: Path, destination: Path, force: bool) -> None:
    """Safely extract the official LoD1 ZIP into a prepared tile directory.

    Args:
        archive: Downloaded official LoD1 ZIP archive.
        destination: Directory that should contain extracted CityGML files.
        force: Replace an existing extracted directory when true.

    Returns:
        None.

    Raises:
        FileNotFoundError: If the archive does not exist.
        DownloadError: If the archive is unsafe, malformed, or unexpectedly large.

    """
    if destination.exists() and any(destination.iterdir()) and not force:
        print(f"[skip] {destination} already contains extracted files")
        return
    if not archive.is_file():
        raise FileNotFoundError(f"LoD1 archive not found: {archive}")

    temporary_directory = destination.with_name(destination.name + ".part")
    shutil.rmtree(temporary_directory, ignore_errors=True)
    temporary_directory.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(archive) as zip_file:
            members = zip_file.infolist()
            total_size = sum(member.file_size for member in members)
            if total_size > MAX_EXTRACTED_BYTES:
                raise DownloadError(f"LoD1 archive expands to {total_size:,} bytes")

            for member in members:
                unix_mode = member.external_attr >> 16
                if stat.S_ISLNK(unix_mode):
                    raise DownloadError(f"Symbolic ZIP member is not allowed: {member.filename}")
                target = _safe_zip_target(temporary_directory, member.filename)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zip_file.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, DOWNLOAD_CHUNK_BYTES)

        if destination.exists():
            shutil.rmtree(destination)
        temporary_directory.replace(destination)
    except (OSError, zipfile.BadZipFile, DownloadError) as error:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise DownloadError(f"Failed to extract LoD1 archive: {archive}") from error

    print(f"[ok] LoD1 archive extracted to {destination}")


def merge_lod1_tiles(tile_directory: Path, output_path: Path, force: bool) -> None:
    """Run the existing CityGML merger on extracted LoD1 files.

    Args:
        tile_directory: Directory containing extracted CityGML source files.
        output_path: Gzipped CityGML file expected by the dataset pipeline.
        force: Replace an existing merged file when true.

    Returns:
        None.

    Raises:
        FileNotFoundError: If no extracted tile directory exists.
        DownloadError: If the merger command fails.

    """
    if output_path.exists() and not force:
        print(f"[skip] {output_path} already exists")
        return
    if not tile_directory.is_dir():
        raise FileNotFoundError(f"LoD1 tile directory not found: {tile_directory}")

    command = [
        sys.executable,
        str(CITYGML_MERGER),
        "--src",
        str(tile_directory),
        "--out",
        str(output_path),
        "--progress",
    ]
    try:
        # The fixed executable and local argument vector contain no shell input.
        subprocess.run(  # noqa: S603 - The argument vector is fixed and shell-free.
            command,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise DownloadError(f"CityGML merger failed with code {error.returncode}") from error


def _write_manifest(input_root: Path, records: list[DownloadRecord]) -> Path:
    """Write source provenance and integrity records as JSON.

    Args:
        input_root: Root directory for downloaded input data.
        records: Completed download records.

    Returns:
        Path to the written manifest.

    Raises:
        OSError: If the manifest cannot be written.

    """
    manifest_path = input_root / "download_manifest.json"
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "license": {"name": LICENSE_NAME, "url": LICENSE_URL},
        "metadata_sources": {
            "lod1_atom": LOD1_ATOM_URL,
            "parcels_catalog": PARCELS_CATALOG_URL,
            "streets_wfs": STREETS_WFS_URL,
        },
        "artifacts": [asdict(record) for record in records],
    }
    input_root.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_suffix(".json.part")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(manifest_path)
    return manifest_path


def _selected_datasets(arguments: argparse.Namespace) -> list[str]:
    """Resolve explicit CLI selections into a stable processing order."""
    selected = set(arguments.datasets or [])
    if arguments.all:
        selected.update(("lod1", "parcels", "streets"))
    if arguments.prepare_lod1:
        selected.add("lod1")
    return [name for name in ("lod1", "parcels", "streets") if name in selected]


def _print_dry_run(selected: list[str], input_root: Path, prepare_lod1: bool) -> None:
    """Print the exact planned sources and destinations without network access."""
    input_directory = input_root / "input"
    if "lod1" in selected:
        print(f"lod1: {LOD1_ARCHIVE_URL} -> {input_directory / 'LoD1.zip'}")
    for name in ("parcels", "streets"):
        if name in selected:
            source = WFS_SOURCES[name]
            print(
                f"{name}: {source.endpoint} [{source.type_name}] -> "
                f"{input_directory / source.filename}"
            )
    if prepare_lod1:
        print(f"extract: {input_directory / 'LoD1.zip'} -> {input_directory / 'lod1_tiles'}")
        print(
            "merge: "
            f"{input_directory / 'lod1_tiles'} -> "
            f"{input_directory / 'berlin_lod1_merged.gml.gz'}"
        )


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for official input-data downloads."""
    parser = argparse.ArgumentParser(
        description="Download official Berlin inputs for the voxel dataset pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 01_CreateDataset/00_download_input_data.py --datasets lod1 parcels
  python 01_CreateDataset/00_download_input_data.py --datasets lod1 parcels --prepare-lod1
  python 01_CreateDataset/00_download_input_data.py --all --dry-run
        """,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("lod1", "parcels", "streets"),
        help="Datasets to download. Streets are optional for the existing pipeline.",
    )
    parser.add_argument("--all", action="store_true", help="Download all three source datasets.")
    parser.add_argument(
        "--prepare-lod1",
        action="store_true",
        help="Download, safely extract, and merge LoD1 into the pipeline input file.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Input-data root (default: {DEFAULT_INPUT_ROOT})",
    )
    parser.add_argument("--page-size", type=int, default=10_000, help="WFS page size (1-50000).")
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument("--retries", type=int, default=3, help="Request attempts (1-10).")
    parser.add_argument("--force", action="store_true", help="Replace existing local artifacts.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without network access.",
    )
    return parser


def main() -> None:
    """Download selected official sources and optionally prepare merged LoD1."""
    parser = _build_parser()
    arguments = parser.parse_args()
    selected = _selected_datasets(arguments)
    if not selected:
        parser.error("select --datasets, --all, or --prepare-lod1")

    try:
        _validate_network_options(arguments.timeout, arguments.retries, arguments.page_size)
    except ValueError as error:
        parser.error(str(error))

    input_root = arguments.input_root.expanduser().resolve()
    input_directory = input_root / "input"
    network = NetworkOptions(
        timeout_seconds=arguments.timeout,
        retries=arguments.retries,
        page_size=arguments.page_size,
    )
    if arguments.dry_run:
        _print_dry_run(selected, input_root, arguments.prepare_lod1)
        return

    records: list[DownloadRecord] = []
    if "lod1" in selected:
        records.append(
            download_binary_file(
                "lod1",
                LOD1_ARCHIVE_URL,
                input_directory / "LoD1.zip",
                network,
                arguments.force,
            )
        )

    for name in ("parcels", "streets"):
        if name not in selected:
            continue
        source = WFS_SOURCES[name]
        records.append(
            download_wfs_geojson(
                source,
                input_directory / source.filename,
                network,
                arguments.force,
            )
        )

    if arguments.prepare_lod1:
        tile_directory = input_directory / "lod1_tiles"
        extract_lod1_archive(input_directory / "LoD1.zip", tile_directory, arguments.force)
        merge_lod1_tiles(
            tile_directory,
            input_directory / "berlin_lod1_merged.gml.gz",
            arguments.force,
        )

    manifest_path = _write_manifest(input_root, records)
    print(f"[ok] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
