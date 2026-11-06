"""Discover, validate, repair, and optionally normalize lumbar-spine DICOM data.

--------------------------------------------------------------------------------------

The input tree is treated as read-only.  DICOM headers, rather than arbitrary
folder names, determine which files belong to each image series.  This is
important because a patient can have multiple series with similar names, and
the filename order is not guaranteed to be the physical slice order.

--------------------------------------------------------------------------------------

Typical usage:

    python preprocessor.py discover --input patients --manifest manifest.csv
    python preprocessor.py repair --input patients --output repaired

--------------------------------------------------------------------------------------

discover:
    - Find all DICOM files
    - Identify the patient they belong to
    - Determine whether they are axial/sagittal/coronal (don't know if there are any) and T1/T2
    - Organize their slices in the correct physical order

repair:
    - Create corrected copies of the DICOM files so that 3DSlicer can interpret and import them more reliably

normalize:
    - Covert selected MRI series into 3D numpy arrays
    - adjusts their spatial resolution
    - standardizes their intensity values for later ML applications

"""

from __future__ import annotations # used for type annotations

import argparse # lets me run commands with terminal args
import csv # used to create the terminal manifest
import importlib # imports optional libraries when needed
import json # used for creating metadata files aloingside the normalized volumes
import logging # warnings/errors
import math
import re # searches text
import sys
from collections import defaultdict # lets me create an empty list when accessing a new key (i.e., patient#1 - nnnnnnnnn)
from dataclasses import dataclass, field # used to build patient objects into a class struct, allows me to then customize how certain class attributes are initialized
from pathlib import Path
from typing import Iterable, Sequence # lets me iterate over an object, lists and tuples

import pydicom # library that lets python read and manipulate DICOMs
from pydicom.dataset import FileDataset # represents the contents of a DICOM file, including metadata and image pixels
from pydicom.dataset import FileMetaDataset # represents the file-level metadata, including how it is encoded and stored
from pydicom.errors import InvalidDicomError # error handling
from pydicom.uid import ExplicitVRLittleEndian, generate_uid # identifies a particular DICOM transfer syntax (how data is encoded), used for creating unique IDs

LOG = logging.getLogger("preprocessor")
# This expression recognizes only standalone T1/T2 tokens, avoiding accidental
# matches inside unrelated words.
CONTRAST_RE = re.compile(r"\b(T1|T2)\b", re.IGNORECASE)


@dataclass
class SliceInfo:
    """Metadata needed to locate and order one DICOM image.

    ``projection`` is the image position projected onto the series normal.
    Sorting by it is more reliable than sorting by filenames or instance
    numbers, which may contain gaps or start at a non-zero value.
    """

    # Original file location; the source dataset is never overwritten.
    path: Path
    # Header-only dataset used during discovery and validation.
    dataset: FileDataset
    # Physical coordinate used for deterministic slice ordering.
    projection: float
    # DICOM's displayed sequence number, retained as a fallback tie-breaker.
    instance_number: int


@dataclass
class Series:
    """Description of one DICOM series and its ordered slices."""

    # Patient and series identifiers come from DICOM, not directory names.
    patient_id: str
    series_uid: str
    # Retained for auditability and troubleshooting.
    source_folder: str
    description: str
    protocol: str
    # Inferred from orientation and contrast metadata respectively.
    view: str
    contrast: str
    # Image geometry needed for resampling and physical measurements.
    rows: int
    columns: int
    pixel_spacing: tuple[float, float]
    slice_thickness: float | None
    spacing_between_slices: float | None
    # These are populated after the series is created.
    slices: list[SliceInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Return a readable label such as ``T2_axial`` for output names."""
        contrast = self.contrast if self.contrast != "unknown" else "unknown"
        view = self.view if self.view != "unknown" else "unknown-view"
        return f"{contrast}_{view}"

    @property
    def sort_key(self) -> tuple[str, str, str]:
        """Return a stable ordering for manifest rows and output folders."""
        return (self.patient_id, self.label, self.series_uid)


def _text(ds: FileDataset, name: str) -> str:
    """Read a DICOM text attribute and convert missing values to ``""``."""
    value = getattr(ds, name, "")
    return str(value).strip() if value is not None else ""


def _float(value: object, default: float | None = None) -> float | None:
    """Convert a DICOM numeric value while handling missing/malformed values."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _orientation(ds: FileDataset) -> tuple[float, ...] | None:
    """Return the six orientation-cosine values when they are valid."""
    try:
        values = tuple(float(v) for v in ds.ImageOrientationPatient)
        return values if len(values) == 6 else None
    except (AttributeError, TypeError, ValueError):
        return None


def infer_view(ds: FileDataset) -> str:
    """Infer axial, coronal, or sagittal orientation from DICOM geometry.

    ``ImageOrientationPatient`` contains direction cosines for the image row
    and column.  Their cross product is the normal to the image plane.  The
    dominant patient axis of that normal identifies the view:

    * normal along X -> sagittal
    * normal along Y -> coronal
    * normal along Z -> axial

    A missing or malformed orientation is reported as ``"unknown"`` rather
    than guessed from a directory name.
    """
    orientation = _orientation(ds)
    if orientation is None:
        return "unknown"
    row, column = orientation[:3], orientation[3:]
    normal = (
        row[1] * column[2] - row[2] * column[1],
        row[2] * column[0] - row[0] * column[2],
        row[0] * column[1] - row[1] * column[0],
    )
    dominant_axis = max(range(3), key=lambda index: abs(normal[index]))
    return ("sagittal", "coronal", "axial")[dominant_axis]


def infer_contrast(ds: FileDataset) -> str:
    """Infer T1/T2 contrast from descriptive DICOM fields.

    This is deliberately conservative.  If no standalone T1 or T2 token is
    found, ``"unknown"`` is returned so a human can review the manifest.
    """
    fields = (_text(ds, "SeriesDescription"), _text(ds, "ProtocolName"),
              _text(ds, "SequenceName"))
    text = " ".join(fields).upper()
    matches = CONTRAST_RE.findall(text)
    return matches[0].upper() if matches else "unknown"


def _projection(ds: FileDataset) -> float:
    """Return a physical coordinate used to order slices in one series.

    The image position is projected onto the series normal.  This makes slice
    ordering work even when filenames are non-sequential.  ``SliceLocation``
    is used only as a fallback when the geometric tags are unavailable.
    """
    try:
        orientation = _orientation(ds)
        position = tuple(float(v) for v in ds.ImagePositionPatient)
        if orientation is None or len(position) != 3:
            return _float(getattr(ds, "SliceLocation", 0), 0) or 0
        row, column = orientation[:3], orientation[3:]
        normal = (
            row[1] * column[2] - row[2] * column[1],
            row[2] * column[0] - row[0] * column[2],
            row[0] * column[1] - row[1] * column[0],
        )
        return sum(a * b for a, b in zip(position, normal))
    except (AttributeError, TypeError, ValueError):
        return 0


def _read_header(path: Path) -> FileDataset | None:
    """Read only DICOM headers so discovery does not load pixel arrays."""
    try:
        return pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except (OSError, InvalidDicomError, ValueError) as exc:
        LOG.warning("Skipping unreadable DICOM %s: %s", path, exc)
        return None


def discover_series(input_root: Path) -> list[Series]:
    """Discover and validate all DICOM series below ``input_root``.

    Files are grouped by ``(PatientID, SeriesInstanceUID)``.  A missing series
    UID receives a synthetic group key so files are not silently discarded,
    while the resulting manifest makes the missing identifier visible.
    """
    # The value stores both the path and header because both are needed later.
    grouped: dict[tuple[str, str], list[tuple[Path, FileDataset]]] = defaultdict(list)
    # rglob supports the nested patient/series layout and future deeper layouts.
    for path in sorted(input_root.rglob("*.dcm")):
        ds = _read_header(path)
        if ds is None:
            continue
        # PatientID is authoritative; the path fallback supports malformed data.
        patient_id = _text(ds, "PatientID") or path.parts[-3]
        series_uid = _text(ds, "SeriesInstanceUID")
        if not series_uid:
            series_uid = f"missing-uid-{path.parent}"
        grouped[(patient_id, series_uid)].append((path, ds))

    result: list[Series] = []
    for (patient_id, series_uid), files in grouped.items():
        # One representative header supplies series-level geometry and labels.
        first = files[0][1]
        view = infer_view(first)
        rows = int(getattr(first, "Rows", 0) or 0)
        columns = int(getattr(first, "Columns", 0) or 0)
        try:
            pixel_spacing = tuple(float(v) for v in first.PixelSpacing)
            if len(pixel_spacing) != 2:
                raise ValueError
        except (AttributeError, TypeError, ValueError):
            # NaN preserves the fact that spacing was unavailable.
            pixel_spacing = (math.nan, math.nan)
        series = Series(
            patient_id=patient_id,
            series_uid=series_uid,
            source_folder=str(files[0][0].parent),
            description=_text(first, "SeriesDescription"),
            protocol=_text(first, "ProtocolName"),
            view=view,
            contrast=infer_contrast(first),
            rows=rows,
            columns=columns,
            pixel_spacing=pixel_spacing,
            slice_thickness=_float(getattr(first, "SliceThickness", None)),
            spacing_between_slices=_float(getattr(first, "SpacingBetweenSlices", None)),
        )
        for path, ds in files:
            # These checks flag inconsistent data without silently dropping it.
            if infer_view(ds) != view:
                series.warnings.append("inconsistent image orientation")
            if (int(getattr(ds, "Rows", 0) or 0), int(getattr(ds, "Columns", 0) or 0)) != (rows, columns):
                series.warnings.append("inconsistent image dimensions")
            series.slices.append(
                SliceInfo(path, ds, _projection(ds), int(_float(getattr(ds, "InstanceNumber", 0), 0) or 0))
            )
        # Physical position is primary; instance number and path make ties stable.
        series.slices.sort(key=lambda item: (item.projection, item.instance_number, str(item.path)))
        projections = [round(item.projection, 4) for item in series.slices]
        if len(projections) != len(set(projections)):
            series.warnings.append("duplicate slice positions")
        # Sort the completed series list so repeated runs produce identical output.
        result.append(series)
    return sorted(result, key=lambda item: item.sort_key)


def _manifest_row(series: Series) -> dict[str, object]:
    """Convert a :class:`Series` into one CSV-compatible dictionary row."""
    return {
        "patient_id": series.patient_id,
        "series_uid": series.series_uid,
        "label": series.label,
        "view": series.view,
        "contrast": series.contrast,
        "description": series.description,
        "protocol": series.protocol,
        "file_count": len(series.slices),
        "rows": series.rows,
        "columns": series.columns,
        "pixel_spacing": "x".join(str(v) for v in series.pixel_spacing),
        "slice_thickness": series.slice_thickness,
        "spacing_between_slices": series.spacing_between_slices,
        "source_folder": series.source_folder,
        "warnings": "; ".join(sorted(set(series.warnings))),
    }


def write_manifest(series: Iterable[Series], destination: Path) -> None:
    """Write one audit row per discovered series to ``destination``."""
    # Materialize the iterable once because it may be a generator.
    rows = [_manifest_row(item) for item in series]
    # Create parent folders so callers can choose nested output paths.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        # Keep a fixed schema, including when the input contains no series.
        fieldnames = [
            "patient_id", "series_uid", "label", "view", "contrast",
            "description", "protocol", "file_count", "rows", "columns",
            "pixel_spacing", "slice_thickness", "spacing_between_slices",
            "source_folder", "warnings",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_id(value: str) -> str:
    """Make a DICOM-derived identifier safe for use as a folder name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def repair_series(series: Series, output_root: Path, study_uid: str) -> Path:
    """Write a repaired, ordered copy of ``series`` and return its folder.

    The repair addresses viewer/import problems by supplying standard
    patient/study/series descriptions and writing a complete DICOM file
    format.  New UIDs distinguish the repaired copy from the source series.
    """
    # Keep patient folders separate and retain a short UID suffix for tracing.
    target = output_root / _safe_id(series.patient_id) / (
        f"{series.label}_{_safe_id(series.series_uid[-12:])}"
    )
    target.mkdir(parents=True, exist_ok=True)
    # All slices in this repaired output belong to one new series.
    repaired_series_uid = generate_uid()
    for index, info in enumerate(series.slices, start=1):
        # Read pixel data and the complete dataset only when making a copy.
        ds = pydicom.dcmread(info.path, force=True)
        # Populate identifiers that some viewers require for a valid patient.
        ds.PatientID = series.patient_id
        ds.PatientName = f"Patient_{series.patient_id}"
        # Give the study and series meaningful labels in 3D Slicer.
        ds.StudyDescription = "Lumbar spine MRI"
        ds.SeriesDescription = f"{series.contrast} {series.view}"
        ds.Modality = "MR"
        # Use the same study but a new series UID for this repaired series.
        ds.SeriesInstanceUID = repaired_series_uid
        ds.StudyInstanceUID = study_uid
        ds.SeriesNumber = 1
        # Replace gaps such as 18.dcm starting a series with 1, 2, 3, ...
        ds.InstanceNumber = index
        ds.ImageComments = "Series identified and ordered by preprocessor.py"
        # Ensure a file-meta dataset exists before setting transfer syntax.
        if not getattr(ds, "file_meta", None):
            ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = getattr(
            ds.file_meta, "TransferSyntaxUID", None
        ) or ExplicitVRLittleEndian
        # Link the file metadata to the stored SOP instance when available.
        if hasattr(ds, "SOPInstanceUID"):
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        # enforce_file_format writes the File Meta Information header/group length.
        ds.save_as(target / f"{index:04d}.dcm", enforce_file_format=True)
    return target


def _normalise_volume(series: Series, output: Path, target_spacing: tuple[float, float, float],
                      lower: float, upper: float) -> None:
    """Resample and intensity-normalize one series into a NumPy volume.

    The output axis order is ``(slice, row, column)``.  Intensities are clipped
    to the requested percentile range, resampled with cubic B-spline-like
    interpolation through SciPy, and z-score normalized.  This is a useful
    model-input baseline, not a substitute for validating an imaging protocol.
    """
    try:
        # Lazy imports keep discovery/repair usable without numerical packages.
        np = importlib.import_module("numpy")
        zoom = importlib.import_module("scipy.ndimage").zoom
    except ImportError as exc:
        raise RuntimeError(
            "normalize requires numpy and scipy; install them in the project environment"
        ) from exc
    # Pixel data is loaded only for the selected series.
    datasets = [pydicom.dcmread(info.path, force=True) for info in series.slices]
    arrays = [ds.pixel_array.astype(np.float32) for ds in datasets]
    volume = np.stack(arrays)
    # DICOM stores in-plane spacing separately from through-plane spacing.
    source_spacing = (
        float(series.spacing_between_slices or series.slice_thickness or 1),
        *series.pixel_spacing,
    )
    # zoom factors convert each source voxel dimension to target spacing.
    factors = tuple(source / target for source, target in zip(source_spacing, target_spacing))
    volume = zoom(volume, factors, order=3)
    # Percentile clipping reduces scanner-dependent outlier influence.
    lo, hi = np.percentile(volume, (lower, upper))
    volume = np.clip(volume, lo, hi)
    # Z-score normalization puts each volume on a comparable intensity scale.
    mean, std = float(volume.mean()), float(volume.std())
    if std > 0:
        volume = (volume - mean) / std
    # Store values and provenance separately so preprocessing is reproducible.
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, volume.astype(np.float32))
    output.with_suffix(".json").write_text(json.dumps({
        "patient_id": series.patient_id,
        "series_uid": series.series_uid,
        "label": series.label,
        "source_spacing_mm": source_spacing,
        "target_spacing_mm": target_spacing,
        "clip_percentiles": [lower, upper],
        "z_score_mean": mean,
        "z_score_std": std,
        "array_order": "slice, row, column",
    }, indent=2) + "\n", encoding="utf-8")


def _parse_spacing(value: str) -> tuple[float, float, float]:
    """Parse ``slice,row,column`` spacing supplied on the command line."""
    parts = tuple(float(part) for part in value.split(","))
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("spacing must be three positive comma-separated values")
    return parts


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and execute the selected pipeline stage."""
    # The module docstring becomes the command's help text.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("discover", "repair", "normalize"):
        # Shared options are defined on every subcommand for convenient usage.
        sub = subparsers.add_parser(command)
        sub.add_argument("--input", type=Path, default=Path("patients"))
        sub.add_argument("--manifest", type=Path, default=Path("manifest.csv"))
        sub.add_argument("--verbose", action="store_true")
    repair = subparsers.choices["repair"]
    repair.add_argument("--output", type=Path, default=Path("repaired"))
    normalize = subparsers.choices["normalize"]
    normalize.add_argument("--output", type=Path, default=Path("normalized"))
    normalize.add_argument("--target-spacing", type=_parse_spacing, default=(5.0, 0.5, 0.5))
    normalize.add_argument("--clip-low", type=float, default=0.05)
    normalize.add_argument("--clip-high", type=float, default=99.5)
    args = parser.parse_args(argv)
    # INFO output is useful for batch runs; warnings remain visible by default.
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")

    # Fail before scanning if the input path is invalid.
    if not args.input.is_dir():
        parser.error(f"input directory does not exist: {args.input}")
    # Every command first discovers series and writes an audit manifest.
    series = discover_series(args.input)
    write_manifest(series, args.manifest)
    LOG.info("Discovered %d series; manifest written to %s", len(series), args.manifest)
    if args.command == "repair":
        # Reuse one study UID for all repaired series belonging to one patient.
        studies: dict[str, str] = {}
        for item in series:
            studies.setdefault(item.patient_id, generate_uid())
            repair_series(item, args.output, studies[item.patient_id])
    elif args.command == "normalize":
        try:
            for item in series:
                # Include patient, label, and UID suffix to prevent collisions.
                output = args.output / _safe_id(item.patient_id) / f"{item.label}_{_safe_id(item.series_uid[-12:])}.npy"
                _normalise_volume(item, output, args.target_spacing, args.clip_low, args.clip_high)
        except RuntimeError as exc:
            # Convert an optional-dependency failure into a useful CLI message.
            LOG.error("%s", exc)
            return 2
    return 0


if __name__ == "__main__":
    # Propagate the command's exit status to the operating system.
    sys.exit(main())
