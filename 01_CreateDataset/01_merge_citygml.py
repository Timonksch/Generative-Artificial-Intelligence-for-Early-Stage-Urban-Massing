#!/usr/bin/env python3
"""Merge CityGML 1.0/2.0 tiles into one constant-memory XML stream."""

from __future__ import annotations

import argparse
import copy
import gzip
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final

from lxml import etree

NAMESPACES: Final = {
    "gml": "http://www.opengis.net/gml",
    "core1": "http://www.opengis.net/citygml/1.0",
    "core2": "http://www.opengis.net/citygml/2.0",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
    "xlink": "http://www.w3.org/1999/xlink",
}
CITY_OBJECT_MEMBER_TAGS: Final = tuple(
    f"{{{NAMESPACES[key]}}}cityObjectMember" for key in ("core1", "core2")
)
GML_ENVELOPE_TAG: Final = f"{{{NAMESPACES['gml']}}}Envelope"
GML_ID_ATTRIBUTE: Final = f"{{{NAMESPACES['gml']}}}id"
XLINK_HREF_ATTRIBUTE: Final = f"{{{NAMESPACES['xlink']}}}href"
SUPPORTED_OUTPUT_SUFFIXES: Final = (".gml", ".xml", ".gml.gz")
MAX_PROMPT_ATTEMPTS: Final = 10
MIN_CORNER_COORDINATES: Final = 2
Envelope = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class StreamMetadata:
    """Store document-level values required by the XML writer."""

    core_namespace: str
    envelope: Envelope | None
    srs_name: str | None


@contextmanager
def _open_input(path: Path) -> Iterator[BinaryIO]:
    """Open compressed or plain CityGML input.

    Args:
        path: Input file path.

    Yields:
        Binary input stream.

    """
    if path.suffix.casefold() == ".gz":
        with gzip.open(path, "rb") as handle:
            yield handle
        return
    with path.open("rb") as handle:
        yield handle


@contextmanager
def _open_output(path: Path, *, compressed: bool) -> Iterator[BinaryIO]:
    """Open a binary output stream.

    Args:
        path: Temporary output path.
        compressed: Write a deterministic gzip stream when true.

    Yields:
        Binary output stream.

    """
    if compressed:
        with (
            path.open("wb") as raw_handle,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as handle,
        ):
            yield handle
        return
    with path.open("wb") as handle:
        yield handle


def _is_citygml_path(path: Path, *, include_gz: bool) -> bool:
    """Check whether a path has a supported CityGML input suffix.

    Args:
        path: Candidate path.
        include_gz: Accept ``.gml.gz`` when true.

    Returns:
        Whether the candidate is supported.

    """
    name = path.name.casefold()
    return name.endswith((".gml", ".xml")) or (include_gz and name.endswith(".gml.gz"))


def glob_inputs(src: Path, include_gz: bool = True) -> list[Path]:
    """Discover CityGML files in stable path order.

    Args:
        src: Source file or recursively searched directory.
        include_gz: Include gzip-compressed GML files.

    Returns:
        Sorted input paths.

    Raises:
        FileNotFoundError: If ``src`` does not exist.
        ValueError: If a single input file has an unsupported suffix.

    """
    if not src.exists():
        raise FileNotFoundError(f"CityGML source not found: {src}")
    if src.is_file():
        if not _is_citygml_path(src, include_gz=include_gz):
            raise ValueError(f"Unsupported CityGML input suffix: {src}")
        return [src]
    return sorted(
        path
        for path in src.rglob("*")
        if path.is_file() and _is_citygml_path(path, include_gz=include_gz)
    )


def detect_core_ns(path: Path) -> str:
    """Detect the CityGML core namespace from the root element.

    Args:
        path: CityGML input file.

    Returns:
        CityGML 1.0 or 2.0 core namespace URI.

    Raises:
        ValueError: If the root does not use a supported core namespace.
        etree.XMLSyntaxError: If the XML is malformed.

    """
    with _open_input(path) as handle:
        context = etree.iterparse(
            handle,
            events=("start",),
            huge_tree=True,
            no_network=True,
            resolve_entities=False,
        )
        _event, root = next(context)
        namespace = etree.QName(root).namespace
    if namespace not in {NAMESPACES["core1"], NAMESPACES["core2"]}:
        raise ValueError(f"Unsupported CityGML root namespace in {path}: {namespace!r}")
    return namespace


def iter_members(path: Path) -> Iterator[etree._Element]:
    """Yield detached CityGML object members while releasing parsed XML.

    Args:
        path: CityGML input file.

    Yields:
        Independent ``cityObjectMember`` elements.

    Raises:
        etree.XMLSyntaxError: If the XML is malformed.

    """
    with _open_input(path) as handle:
        context = etree.iterparse(
            handle,
            events=("end",),
            tag=CITY_OBJECT_MEMBER_TAGS,
            huge_tree=True,
            no_network=True,
            resolve_entities=False,
        )
        for _event, element in context:
            yield copy.deepcopy(element)
            element.clear()
            parent = element.getparent()
            while parent is not None and element.getprevious() is not None:
                del parent[0]


def rewrite_ids_and_hrefs(element: etree._Element, prefix: str) -> None:
    """Prefix GML IDs and local XLink references consistently.

    Args:
        element: CityGML object member modified in place.
        prefix: Unique XML-ID-safe tile prefix.

    """
    for descendant in element.iter():
        identifier = descendant.get(GML_ID_ATTRIBUTE)
        if identifier:
            descendant.set(GML_ID_ATTRIBUTE, f"{prefix}__{identifier}")
        href = descendant.get(XLINK_HREF_ATTRIBUTE)
        if href and href.startswith("#"):
            descendant.set(XLINK_HREF_ATTRIBUTE, f"#{prefix}__{href[1:]}")


def _parse_corner(envelope: etree._Element, name: str) -> tuple[float, float] | None:
    """Parse the horizontal coordinates of one GML envelope corner.

    Args:
        envelope: GML ``Envelope`` element.
        name: ``lowerCorner`` or ``upperCorner``.

    Returns:
        Horizontal coordinate pair, or ``None`` when absent.

    Raises:
        ValueError: If fewer than two numeric coordinates are present.

    """
    element = envelope.find(f"{{{NAMESPACES['gml']}}}{name}")
    if element is None or not element.text:
        return None
    values = [float(value) for value in element.text.split()]
    if len(values) < MIN_CORNER_COORDINATES:
        raise ValueError(f"GML {name} must contain at least two coordinates")
    return values[0], values[1]


def read_file_envelope_and_srs(path: Path) -> tuple[Envelope | None, str | None]:
    """Read the first GML envelope without loading the complete tile.

    Args:
        path: CityGML input file.

    Returns:
        Horizontal envelope and optional SRS identifier.

    Raises:
        ValueError: If a present envelope is malformed.
        etree.XMLSyntaxError: If the XML is malformed before the envelope.

    """
    with _open_input(path) as handle:
        context = etree.iterparse(
            handle,
            events=("end",),
            tag=GML_ENVELOPE_TAG,
            huge_tree=True,
            no_network=True,
            resolve_entities=False,
        )
        for _event, element in context:
            lower = _parse_corner(element, "lowerCorner")
            upper = _parse_corner(element, "upperCorner")
            if lower is None or upper is None:
                return None, element.get("srsName")
            return (lower[0], lower[1], upper[0], upper[1]), element.get("srsName")
    return None, None


def envelope_union(first: Envelope | None, second: Envelope | None) -> Envelope | None:
    """Compute the union of two horizontal envelopes.

    Args:
        first: First envelope or ``None``.
        second: Second envelope or ``None``.

    Returns:
        Combined envelope or the existing non-null value.

    """
    if first is None:
        return second
    if second is None:
        return first
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def _collect_global_envelope(files: Sequence[Path]) -> tuple[Envelope | None, str | None]:
    """Collect one envelope and consistent SRS across all input files.

    Args:
        files: CityGML input paths.

    Returns:
        Global envelope and shared SRS.

    Raises:
        ValueError: If files declare conflicting SRS identifiers.

    """
    global_envelope: Envelope | None = None
    global_srs: str | None = None
    for path in files:
        envelope, srs_name = read_file_envelope_and_srs(path)
        global_envelope = envelope_union(global_envelope, envelope)
        if srs_name and global_srs and srs_name != global_srs:
            raise ValueError(f"Conflicting SRS identifiers: {global_srs!r} and {srs_name!r}")
        global_srs = global_srs or srs_name
    return global_envelope, global_srs


def _envelope_element(envelope: Envelope, srs_name: str | None) -> etree._Element:
    """Build a CityModel ``boundedBy`` element.

    Args:
        envelope: Global horizontal bounds.
        srs_name: Optional coordinate reference identifier.

    Returns:
        GML ``boundedBy`` element.

    """
    bounded_by = etree.Element(f"{{{NAMESPACES['gml']}}}boundedBy")
    envelope_element = etree.SubElement(bounded_by, GML_ENVELOPE_TAG)
    if srs_name:
        envelope_element.set("srsName", srs_name)
    lower = etree.SubElement(envelope_element, f"{{{NAMESPACES['gml']}}}lowerCorner")
    lower.text = f"{envelope[0]} {envelope[1]}"
    upper = etree.SubElement(envelope_element, f"{{{NAMESPACES['gml']}}}upperCorner")
    upper.text = f"{envelope[2]} {envelope[3]}"
    return bounded_by


def _safe_prefix(path: Path, index: int) -> str:
    """Create a deterministic, unique, XML-ID-safe tile prefix.

    Args:
        path: Source tile path.
        index: One-based position in the stable input order.

    Returns:
        Safe tile prefix.

    """
    normalized_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", path.stem)
    return f"tile_{index:05d}_{normalized_stem}"


def _write_stream(
    files: Sequence[Path],
    output_path: Path,
    metadata: StreamMetadata,
    *,
    show_progress: bool,
) -> int:
    """Write all object members to one constant-memory CityGML stream.

    Args:
        files: Ordered input tiles.
        output_path: Temporary output path.
        metadata: Document namespace and global envelope.
        show_progress: Print one message per tile.

    Returns:
        Number of written city object members.

    """
    namespace_map = {
        "gml": NAMESPACES["gml"],
        "core": metadata.core_namespace,
        "xsi": NAMESPACES["xsi"],
        "xlink": NAMESPACES["xlink"],
    }
    member_count = 0
    compressed = output_path.name.endswith(".gml.gz.part")
    with (
        _open_output(output_path, compressed=compressed) as handle,
        etree.xmlfile(handle, encoding="UTF-8") as writer,
    ):
        writer.write_declaration()
        root_tag = f"{{{metadata.core_namespace}}}CityModel"
        with writer.element(root_tag, nsmap=namespace_map):
            if metadata.envelope is not None:
                writer.write(_envelope_element(metadata.envelope, metadata.srs_name))
            for index, path in enumerate(files, start=1):
                if show_progress:
                    print(f"[{index}/{len(files)}] {path}")
                prefix = _safe_prefix(path, index)
                for member in iter_members(path):
                    rewrite_ids_and_hrefs(member, prefix)
                    writer.write(member)
                    member_count += 1
    return member_count


def _validate_output_path(path: Path) -> None:
    """Validate a requested output filename.

    Args:
        path: Output path to validate.

    Raises:
        ValueError: If the output suffix is unsupported.

    """
    if not path.name.casefold().endswith(SUPPORTED_OUTPUT_SUFFIXES):
        suffixes = ", ".join(SUPPORTED_OUTPUT_SUFFIXES)
        raise ValueError(f"Output must end with one of: {suffixes}")


def merge_citygml(
    src: Path,
    out: Path,
    include_gz: bool = True,
    show_progress: bool = False,
) -> None:
    """Merge CityGML files atomically and with bounded memory usage.

    Args:
        src: Source file or recursively searched directory.
        out: Output ``.gml``, ``.xml``, or ``.gml.gz`` path.
        include_gz: Include compressed source tiles.
        show_progress: Print tile-level progress.

    Raises:
        FileNotFoundError: If the source does not exist.
        ValueError: If no inputs exist or versions/SRS identifiers conflict.
        OSError: If output cannot be written.
        etree.XMLSyntaxError: If an input file is malformed.

    """
    _validate_output_path(out)
    files = glob_inputs(src, include_gz=include_gz)
    output_resolved = out.expanduser().resolve()
    files = [path for path in files if path.resolve() != output_resolved]
    if not files:
        raise ValueError(f"No CityGML input files found below: {src}")

    core_namespace = detect_core_ns(files[0])
    for path in files[1:]:
        if detect_core_ns(path) != core_namespace:
            raise ValueError("Cannot merge CityGML 1.0 and 2.0 files into one document")
    envelope, srs_name = _collect_global_envelope(files)

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = out.with_name(f"{out.name}.part")
    try:
        metadata = StreamMetadata(core_namespace, envelope, srs_name)
        count = _write_stream(files, temporary_path, metadata, show_progress=show_progress)
        temporary_path.replace(out)
    except (OSError, ValueError, etree.XMLSyntaxError):
        temporary_path.unlink(missing_ok=True)
        raise
    if show_progress:
        print(f"Merge complete: {out} ({len(files)} files, {count} city objects)")


def _interactive_path(text: str, *, must_exist: bool) -> Path:
    """Read a path with a bounded number of attempts.

    Args:
        text: Prompt shown to the user.
        must_exist: Require the entered path to exist.

    Returns:
        Entered path.

    Raises:
        RuntimeError: If no valid path is entered within the attempt limit.

    """
    for _attempt in range(MAX_PROMPT_ATTEMPTS):
        path = Path(input(f"{text}: ").strip()).expanduser()
        if path and (not must_exist or path.exists()):
            return path
        print(f"Invalid path: {path}")
    raise RuntimeError(f"No valid path after {MAX_PROMPT_ATTEMPTS} attempts")


def interactive_mode() -> None:
    """Run the bounded guided merger dialog."""
    source = _interactive_path("Source file or directory", must_exist=True)
    for _attempt in range(MAX_PROMPT_ATTEMPTS):
        output = _interactive_path("Output .gml, .xml, or .gml.gz", must_exist=False)
        try:
            _validate_output_path(output)
            break
        except ValueError as error:
            print(f"ERROR: {error}")
    else:
        raise RuntimeError(f"No valid output after {MAX_PROMPT_ATTEMPTS} attempts")
    include_gz = input("Include .gml.gz inputs? [Y/n]: ").strip().casefold() != "n"
    show_progress = input("Show progress? [Y/n]: ").strip().casefold() != "n"
    merge_citygml(source, output, include_gz, show_progress)


def _build_parser() -> argparse.ArgumentParser:
    """Create the merger argument parser.

    Returns:
        Configured parser.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, help="Source CityGML file or directory.")
    parser.add_argument("--out", type=Path, help="Output .gml, .xml, or .gml.gz file.")
    parser.add_argument("--no-gz", action="store_true", help="Ignore compressed inputs.")
    parser.add_argument("--progress", action="store_true", help="Print tile progress.")
    parser.add_argument("--interactive", action="store_true", help="Run the guided dialog.")
    return parser


def main() -> None:
    """Validate CLI arguments and merge the requested CityGML inputs."""
    parser = _build_parser()
    arguments = parser.parse_args()
    if arguments.interactive:
        interactive_mode()
        return
    if arguments.src is None or arguments.out is None:
        parser.error("--src and --out are required unless --interactive is used")
    try:
        merge_citygml(
            arguments.src.expanduser(),
            arguments.out.expanduser(),
            include_gz=not arguments.no_gz,
            show_progress=arguments.progress,
        )
    except (OSError, ValueError, etree.XMLSyntaxError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
