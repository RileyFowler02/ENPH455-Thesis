"""Discover, audit, repair, and geometry-preserving export of lumbar MRI DICOM.

The source tree is never modified.  The reference CSV files supplied with the
dataset are authoritative for study/series descriptions and label coordinates;
DICOM geometry remains authoritative for physical image reconstruction.

Examples:

    python preprocessor.py discover --input patients --manifest manifest.csv
    python preprocessor.py repair --input patients --output repaired
    python preprocessor.py normalize --input repaired --output normalized

``normalize`` writes NIfTI (RAS affine) plus JSON provenance.  It preserves
raw modality-corrected intensities by default; z-score normalization is an
explicit option for machine-learning experiments and must not replace the
original or repaired DICOM.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.errors import InvalidDicomError
from pydicom.pixels.processing import apply_modality_lut
from pydicom.uid import generate_uid

LOG = logging.getLogger("preprocessor")


@dataclass(frozen=True)
class ReferenceSeries:
    study_id: str
    series_id: str
    description: str


@dataclass
class SliceInfo:
    path: Path
    dataset: FileDataset
    projection: float
    instance_number: int


@dataclass
class Series:
    patient_id: str
    series_uid: str
    study_uid: str
    source_folder: str
    description: str
    protocol: str
    rows: int
    columns: int
    pixel_spacing: tuple[float, float]
    slice_thickness: float | None
    spacing_between_slices: float | None
    reference: ReferenceSeries | None = None
    coordinate_count: int = 0
    part_index: int = 1
    train_labels: dict[str, str] = field(default_factory=dict)
    slices: list[SliceInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Return the dataset description from the reference CSV, not an inference."""
        if self.reference is None:
            return "unreferenced"
        return _safe_id(self.reference.description.lower().replace("/", "-"))

    @property
    def sort_key(self) -> tuple[str, str, str]:
        return self.patient_id, self.label, f"{self.series_uid}:{self.part_index}"


def _text(ds: FileDataset, name: str) -> str:
    value = getattr(ds, name, "")
    return str(value).strip() if value is not None else ""


def _float(value: object, default: float | None = None) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _orientation(ds: FileDataset) -> tuple[float, ...] | None:
    try:
        values = tuple(float(v) for v in ds.ImageOrientationPatient)
    except (AttributeError, TypeError, ValueError):
        return None
    return values if len(values) == 6 and all(math.isfinite(v) for v in values) else None


def _position(ds: FileDataset) -> tuple[float, float, float] | None:
    try:
        values = tuple(float(v) for v in ds.ImagePositionPatient)
    except (AttributeError, TypeError, ValueError):
        return None
    return values if len(values) == 3 and all(math.isfinite(v) for v in values) else None


def _normal(orientation: tuple[float, ...]) -> tuple[float, float, float]:
    row, column = orientation[:3], orientation[3:]
    return (
        row[1] * column[2] - row[2] * column[1],
        row[2] * column[0] - row[0] * column[2],
        row[0] * column[1] - row[1] * column[0],
    )


def _projection(ds: FileDataset) -> float:
    orientation = _orientation(ds)
    position = _position(ds)
    if orientation is not None and position is not None:
        return sum(a * b for a, b in zip(position, _normal(orientation)))
    return _float(getattr(ds, "SliceLocation", None), 0.0) or 0.0


def _read_header(path: Path) -> FileDataset | None:
    try:
        return pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except (OSError, InvalidDicomError, ValueError) as exc:
        LOG.warning("Skipping unreadable DICOM %s: %s", path, exc)
        return None


def _read_reference_series(path: Path) -> dict[tuple[str, str], ReferenceSeries]:
    if not path.is_file():
        LOG.warning("Reference series CSV not found: %s", path)
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            (row["study_id"].strip(), row["series_id"].strip()): ReferenceSeries(
                row["study_id"].strip(), row["series_id"].strip(),
                row["series_description"].strip(),
            )
            for row in csv.DictReader(handle)
            if row.get("study_id") and row.get("series_id")
        }


def _read_coordinate_counts(path: Path) -> dict[tuple[str, str], int]:
    if not path.is_file():
        LOG.warning("Reference coordinate CSV not found: %s", path)
        return {}
    counts: dict[tuple[str, str], int] = defaultdict(int)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("study_id") and row.get("series_id"):
                counts[(row["study_id"].strip(), row["series_id"].strip())] += 1
    return dict(counts)


def _read_train_labels(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        LOG.warning("Training label CSV not found: %s", path)
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            row["study_id"].strip(): {
                key: value.strip() for key, value in row.items()
                if key != "study_id" and value is not None
            }
            for row in csv.DictReader(handle)
            if row.get("study_id")
        }


def _reference_key(patient_id: str, series_uid: str) -> tuple[str, str]:
    """Return the dataset key used by the CSVs for a DICOM series.

    The competition data stores the numeric series directory/identifier, while
    some DICOM exports prepend an implementation-specific root to the UID.
    """
    return patient_id, series_uid


def _reference_for(
    patient_id: str, series_uid: str, references: dict[tuple[str, str], ReferenceSeries],
) -> ReferenceSeries | None:
    candidates = (
        _reference_key(patient_id, series_uid),
        _reference_key(patient_id, series_uid.rsplit(".", 1)[-1]),
    )
    return next((references[key] for key in candidates if key in references), None)


def _coordinate_count_for(
    patient_id: str, series_uid: str, coordinates: dict[tuple[str, str], int],
) -> int:
    candidates = (
        _reference_key(patient_id, series_uid),
        _reference_key(patient_id, series_uid.rsplit(".", 1)[-1]),
    )
    return next((coordinates[key] for key in candidates if key in coordinates), 0)


def discover_series(
    input_root: Path,
    series_descriptions: Path | None = None,
    label_coordinates: Path | None = None,
    train_labels: Path | None = None,
) -> list[Series]:
    """Discover series and attach the supplied reference metadata."""
    references = _read_reference_series(series_descriptions) if series_descriptions else {}
    coordinates = _read_coordinate_counts(label_coordinates) if label_coordinates else {}
    labels = _read_train_labels(train_labels) if train_labels else {}
    grouped: dict[tuple[str, str, tuple[float, ...] | None], list[tuple[Path, FileDataset]]] = defaultdict(list)
    for path in sorted(input_root.rglob("*.dcm")):
        ds = _read_header(path)
        if ds is None:
            continue
        patient_id = _text(ds, "PatientID") or (path.parts[-3] if len(path.parts) >= 3 else "unknown")
        series_uid = _text(ds, "SeriesInstanceUID") or f"missing-uid-{path.parent}"
        orientation = _orientation(ds)
        orientation_key = tuple(round(value, 4) for value in orientation) if orientation else None
        grouped[(patient_id, series_uid, orientation_key)].append((path, ds))

    result: list[Series] = []
    part_numbers: dict[tuple[str, str], int] = defaultdict(int)
    for (patient_id, series_uid, _), files in grouped.items():
        part_numbers[(patient_id, series_uid)] += 1
        first = files[0][1]
        try:
            spacing = tuple(float(v) for v in first.PixelSpacing)
            if len(spacing) != 2 or any(v <= 0 or not math.isfinite(v) for v in spacing):
                raise ValueError
            pixel_spacing = spacing
        except (AttributeError, TypeError, ValueError):
            pixel_spacing = (math.nan, math.nan)
        reference = _reference_for(patient_id, series_uid, references)
        series = Series(
            patient_id=patient_id,
            series_uid=series_uid,
            study_uid=_text(first, "StudyInstanceUID"),
            source_folder=str(files[0][0].parent),
            description=_text(first, "SeriesDescription"),
            protocol=_text(first, "ProtocolName"),
            rows=int(getattr(first, "Rows", 0) or 0),
            columns=int(getattr(first, "Columns", 0) or 0),
            pixel_spacing=pixel_spacing,
            slice_thickness=_float(getattr(first, "SliceThickness", None)),
            spacing_between_slices=_float(getattr(first, "SpacingBetweenSlices", None)),
            reference=reference,
            coordinate_count=_coordinate_count_for(patient_id, series_uid, coordinates),
            part_index=part_numbers[(patient_id, series_uid)],
            train_labels=labels.get(patient_id, {}),
        )
        if reference is None:
            series.warnings.append("series not present in reference CSV")
        for path, ds in files:
            if (int(getattr(ds, "Rows", 0) or 0), int(getattr(ds, "Columns", 0) or 0)) != (
                series.rows, series.columns
            ):
                series.warnings.append("inconsistent image dimensions")
            try:
                ps = tuple(float(v) for v in ds.PixelSpacing)
                if ps != series.pixel_spacing:
                    series.warnings.append("inconsistent pixel spacing")
            except (AttributeError, TypeError, ValueError):
                series.warnings.append("missing pixel spacing")
            series.slices.append(SliceInfo(
                path, ds, _projection(ds),
                int(_float(getattr(ds, "InstanceNumber", 0), 0) or 0),
            ))
        series.slices.sort(key=lambda item: (item.projection, item.instance_number, str(item.path)))
        projections = [item.projection for item in series.slices]
        if len(projections) > 1:
            gaps = [abs(b - a) for a, b in zip(projections, projections[1:]) if abs(b - a) > 1e-6]
            if len(set(round(gap, 3) for gap in gaps)) > 1:
                series.warnings.append("nonuniform slice spacing")
        if len({round(value, 4) for value in projections}) != len(projections):
            series.warnings.append("duplicate slice positions")
        result.append(series)
    return sorted(result, key=lambda item: item.sort_key)


def _manifest_row(series: Series) -> dict[str, object]:
    positions = [_position(info.dataset) for info in series.slices]
    valid_positions = [position for position in positions if position is not None]
    projections = [info.projection for info in series.slices]
    gaps = [abs(b - a) for a, b in zip(projections, projections[1:]) if abs(b - a) > 1e-6]
    return {
        "patient_id": series.patient_id,
        "study_uid": series.study_uid,
        "series_uid": series.series_uid,
        "geometry_part": series.part_index,
        "reference_series_id": series.reference.series_id if series.reference else "",
        "reference_description": series.reference.description if series.reference else "",
        "label_source": "reference_csv" if series.reference else "unreferenced",
        "coordinate_count": series.coordinate_count,
        "train_labels": json.dumps(series.train_labels, sort_keys=True),
        "label": series.label,
        "dicom_description": series.description,
        "protocol": series.protocol,
        "file_count": len(series.slices),
        "rows": series.rows,
        "columns": series.columns,
        "pixel_spacing_mm": json.dumps(series.pixel_spacing),
        "slice_positions_patient_mm": json.dumps(valid_positions),
        "slice_projections_mm": json.dumps(projections),
        "slice_gap_min_mm": min(gaps) if gaps else "",
        "slice_gap_max_mm": max(gaps) if gaps else "",
        "slice_thickness_mm": series.slice_thickness or "",
        "source_folder": series.source_folder,
        "warnings": "; ".join(sorted(set(series.warnings))),
    }


def write_manifest(series: Iterable[Series], destination: Path) -> None:
    rows = [_manifest_row(item) for item in series]
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "patient_id", "study_uid", "series_uid", "geometry_part", "reference_series_id",
        "reference_description", "label_source", "coordinate_count", "train_labels", "label",
        "dicom_description", "protocol", "file_count", "rows", "columns",
        "pixel_spacing_mm", "slice_positions_patient_mm", "slice_projections_mm",
        "slice_gap_min_mm", "slice_gap_max_mm", "slice_thickness_mm",
        "source_folder", "warnings",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_slice_manifest(series: Iterable[Series], destination: Path) -> None:
    """Write one row per source slice for audit and manual geometry review."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["patient_id", "study_uid", "series_uid", "geometry_part", "reference_description",
              "slice_index", "source_file", "instance_number", "projection_mm",
              "image_position_patient", "image_orientation_patient", "sop_instance_uid"]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in series:
            for index, info in enumerate(item.slices):
                writer.writerow({
                    "patient_id": item.patient_id, "study_uid": item.study_uid,
                    "series_uid": item.series_uid,
                    "geometry_part": item.part_index,
                    "reference_description": item.reference.description if item.reference else "",
                    "slice_index": index, "source_file": str(info.path),
                    "instance_number": info.instance_number,
                    "projection_mm": info.projection,
                    "image_position_patient": json.dumps(_position(info.dataset)),
                    "image_orientation_patient": json.dumps(_orientation(info.dataset)),
                    "sop_instance_uid": _text(info.dataset, "SOPInstanceUID"),
                })


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def repair_series(series: Series, output_root: Path) -> Path:
    """Write a viewer-compatible copy while preserving source study grouping."""
    target = output_root / _safe_id(series.patient_id) / (
        f"{series.label}_{_safe_id(series.series_uid[-12:])}_part{series.part_index}"
    )
    target.mkdir(parents=True, exist_ok=True)
    repaired_series_uid = generate_uid()
    for index, info in enumerate(series.slices, start=1):
        ds = pydicom.dcmread(info.path, force=True)
        ds.PatientID = series.patient_id
        ds.PatientName = f"Patient_{series.patient_id}"
        ds.Modality = _text(ds, "Modality") or "MR"
        ds.SeriesInstanceUID = repaired_series_uid
        if not _text(ds, "StudyInstanceUID"):
            ds.StudyInstanceUID = generate_uid()
        ds.SeriesDescription = series.reference.description if series.reference else series.description
        ds.SeriesNumber = int(_float(getattr(ds, "SeriesNumber", 1), 1) or 1)
        ds.InstanceNumber = index
        ds.ImageComments = "Ordered copy created by preprocessor.py; source study relationship preserved"
        if not getattr(ds, "file_meta", None):
            ds.file_meta = FileMetaDataset()
        # Preserve the source transfer syntax.  No pixel conversion occurs here.
        transfer_syntax = getattr(info.dataset.file_meta, "TransferSyntaxUID", None)
        if transfer_syntax is not None:
            ds.file_meta.TransferSyntaxUID = transfer_syntax
        ds.SOPInstanceUID = generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        if _text(ds, "SOPClassUID"):
            ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        ds.save_as(target / f"{index:04d}.dcm", enforce_file_format=True)
    return target


def _metadata_path(output: Path) -> Path:
    name = output.name[:-7] if output.name.endswith(".nii.gz") else output.stem
    return output.with_name(name + ".json")


def _geometry(series: Series) -> tuple[tuple[float, ...], tuple[float, float, float], list[float]]:
    if not series.slices or any(
        (info.dataset.Rows, info.dataset.Columns) != (series.rows, series.columns)
        for info in series.slices
    ):
        raise ValueError("all slices must have identical, valid dimensions")
    orientations = [_orientation(info.dataset) for info in series.slices]
    positions = [_position(info.dataset) for info in series.slices]
    if any(value is None for value in orientations + positions):
        raise ValueError("ImageOrientationPatient and ImagePositionPatient are required")
    orientation = orientations[0]
    assert orientation is not None
    if any(max(abs(a - b) for a, b in zip(value, orientation)) > 1e-4 for value in orientations[1:] if value):
        raise ValueError("slice orientations differ")
    if any(not math.isfinite(value) or value <= 0 for value in series.pixel_spacing):
        raise ValueError("valid PixelSpacing is required")
    projections = [info.projection for info in series.slices]
    gaps = [abs(b - a) for a, b in zip(projections, projections[1:]) if abs(b - a) > 1e-6]
    if not gaps:
        raise ValueError("at least two distinct slice positions are required")
    spacing = (sum(gaps) / len(gaps), series.pixel_spacing[0], series.pixel_spacing[1])
    return orientation, spacing, projections


def _normalise_volume(
    series: Series, output: Path, target_spacing: tuple[float, float, float],
    lower: float, upper: float, intensity_mode: str, interpolation_order: int,
) -> None:
    """Resample to NIfTI while retaining an affine and the original slice geometry."""
    if not 0 <= lower < upper <= 100:
        raise ValueError("clip percentiles must satisfy 0 <= low < high <= 100")
    if any(value <= 0 or not math.isfinite(value) for value in target_spacing):
        raise ValueError("target spacing must contain three positive finite values")
    np = importlib.import_module("numpy")
    zoom = importlib.import_module("scipy.ndimage").zoom
    nib = importlib.import_module("nibabel")
    orientation, source_spacing, projections = _geometry(series)
    datasets = [pydicom.dcmread(info.path, force=True) for info in series.slices]
    arrays = []
    for ds in datasets:
        array = ds.pixel_array
        try:
            array = apply_modality_lut(array, ds)
        except (AttributeError, ValueError):
            slope = _float(getattr(ds, "RescaleSlope", 1), 1) or 1
            intercept = _float(getattr(ds, "RescaleIntercept", 0), 0) or 0
            array = array * slope + intercept
        arrays.append(np.asarray(array, dtype=np.float32))
    volume = np.stack(arrays)
    factors = tuple(source / target for source, target in zip(source_spacing, target_spacing))
    volume = zoom(volume, factors, order=interpolation_order)
    lo, hi = np.percentile(volume, (lower, upper))
    volume = np.clip(volume, lo, hi)
    mean, std = float(volume.mean()), float(volume.std())
    if intensity_mode == "zscore" and std > 0:
        volume = (volume - mean) / std

    first_position = _position(datasets[0])
    assert first_position is not None
    row, column = orientation[:3], orientation[3:]
    normal = _normal(orientation)
    # DICOM array axes are (slice, row, column); convert the LPS affine to RAS.
    lps = np.eye(4, dtype=float)
    lps[:3, 0] = np.asarray(normal) * target_spacing[0]
    lps[:3, 1] = np.asarray(column) * target_spacing[1]
    lps[:3, 2] = np.asarray(row) * target_spacing[2]
    lps[:3, 3] = first_position
    ras = np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume.astype(np.float32), ras), str(output))
    _metadata_path(output).write_text(json.dumps({
        "patient_id": series.patient_id,
        "study_uid": series.study_uid,
        "series_uid": series.series_uid,
        "reference_description": series.reference.description if series.reference else None,
        "source_files": [str(info.path) for info in series.slices],
        "array_order": "slice, row, column",
        "coordinate_system": "NIfTI RAS; source DICOM tags are LPS",
        "image_position_patient_first_mm": first_position,
        "image_orientation_patient": orientation,
        "slice_positions_patient_mm": [_position(ds) for ds in datasets],
        "slice_projections_mm": projections,
        "source_spacing_mm": source_spacing,
        "target_spacing_mm": target_spacing,
        "nonuniform_slice_spacing": "nonuniform slice spacing" in series.warnings,
        "intensity_mode": intensity_mode,
        "clip_percentiles": [lower, upper],
        "clip_values": [float(lo), float(hi)],
        "modality_lut_applied": True,
        "interpolation_order": interpolation_order,
    }, indent=2) + "\n", encoding="utf-8")


def _parse_spacing(value: str) -> tuple[float, float, float]:
    try:
        parts = tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("spacing must be three positive comma-separated values") from exc
    if len(parts) != 3 or any(part <= 0 or not math.isfinite(part) for part in parts):
        raise argparse.ArgumentTypeError("spacing must be three positive comma-separated values")
    return parts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("discover", "repair", "normalize"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--input", type=Path, default=Path("patients"))
        sub.add_argument("--manifest", type=Path, default=Path("manifest.csv"))
        sub.add_argument("--series-descriptions", type=Path, default=Path("train_series_descriptions.csv"))
        sub.add_argument("--label-coordinates", type=Path, default=Path("train_label_coordinates.csv"))
        sub.add_argument("--labels", type=Path, default=Path("train.csv"))
        sub.add_argument("--verbose", action="store_true")
    repair = subparsers.choices["repair"]
    repair.add_argument("--output", type=Path, default=Path("repaired"))
    normalize = subparsers.choices["normalize"]
    normalize.add_argument("--output", type=Path, default=Path("normalized"))
    normalize.add_argument("--target-spacing", type=_parse_spacing, default=(5.0, 0.5, 0.5))
    normalize.add_argument("--clip-low", type=float, default=0.05)
    normalize.add_argument("--clip-high", type=float, default=99.5)
    normalize.add_argument("--intensity-mode", choices=("raw", "zscore"), default="raw")
    normalize.add_argument("--interpolation-order", type=int, choices=(0, 1, 3), default=1)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s: %(message)s")
    if not args.input.is_dir():
        parser.error(f"input directory does not exist: {args.input}")
    try:
        series = discover_series(args.input, args.series_descriptions, args.label_coordinates, args.labels)
        write_manifest(series, args.manifest)
        write_slice_manifest(series, args.manifest.with_name(args.manifest.stem + "_slices.csv"))
        LOG.info("Discovered %d series; manifests written to %s", len(series), args.manifest)
        if args.command == "repair":
            for item in series:
                repair_series(item, args.output)
        elif args.command == "normalize":
            for item in series:
                output = args.output / _safe_id(item.patient_id) / (
                    f"{item.label}_{_safe_id(item.series_uid[-12:])}_part{item.part_index}.nii.gz"
                )
                _normalise_volume(item, output, args.target_spacing, args.clip_low,
                                  args.clip_high, args.intensity_mode, args.interpolation_order)
    except (ImportError, RuntimeError, ValueError) as exc:
        LOG.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
