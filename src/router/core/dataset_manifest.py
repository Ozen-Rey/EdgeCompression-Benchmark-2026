"""Dataset manifest schema, validation helpers, and CLI."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from src.router.core.domain_spec import DomainSpec, resolve_domain_spec

VALID_DOMAINS = {"image", "audio", "video"}
DATASET_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class DatasetItem:
    item_id: str
    path: str
    metadata: dict[str, Any]
    width: int | None
    height: int | None
    pixels: int | None
    duration_s: float | None
    sample_rate: int | None
    channels: int | None
    fps: float | None
    num_frames: int | None


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: str
    dataset_id: str
    display_name: str | None
    domain: str
    root: str
    items: list[DatasetItem]
    splits: dict[str, list[str]]
    metadata: dict[str, Any]
    license: str | None
    source_url: str | None
    notes: str | None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _coerce_item(data: DatasetItem | Mapping[str, Any]) -> DatasetItem:
    if isinstance(data, DatasetItem):
        return data

    width = _optional_int(data.get("width"))
    height = _optional_int(data.get("height"))
    pixels = _optional_int(data.get("pixels"))
    if pixels is None and width is not None and height is not None:
        pixels = width * height

    metadata = data.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        metadata = {}

    return DatasetItem(
        item_id=str(data.get("item_id", "")),
        path=str(data.get("path", "")),
        metadata=dict(metadata),
        width=width,
        height=height,
        pixels=pixels,
        duration_s=_optional_float(data.get("duration_s")),
        sample_rate=_optional_int(data.get("sample_rate")),
        channels=_optional_int(data.get("channels")),
        fps=_optional_float(data.get("fps")),
        num_frames=_optional_int(data.get("num_frames")),
    )


def normalize_dataset_manifest(
    manifest: DatasetManifest | Mapping[str, Any],
) -> DatasetManifest:
    if isinstance(manifest, DatasetManifest):
        items = [_coerce_item(item) for item in manifest.items]
        splits = {name: list(ids) for name, ids in manifest.splits.items()}
        metadata = dict(manifest.metadata)
        return DatasetManifest(
            schema_version=manifest.schema_version,
            dataset_id=manifest.dataset_id,
            display_name=manifest.display_name,
            domain=manifest.domain,
            root=manifest.root,
            items=items,
            splits=splits,
            metadata=metadata,
            license=manifest.license,
            source_url=manifest.source_url,
            notes=manifest.notes,
        )

    items = [_coerce_item(item) for item in manifest.get("items", [])]
    splits_raw = manifest.get("splits", {}) or {}
    splits = {
        str(split_name): [str(item_id) for item_id in item_ids]
        for split_name, item_ids in dict(splits_raw).items()
    }
    metadata = manifest.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {}

    return DatasetManifest(
        schema_version=str(manifest.get("schema_version", "")),
        dataset_id=str(manifest.get("dataset_id", "")),
        display_name=manifest.get("display_name"),
        domain=str(manifest.get("domain", "")),
        root=str(manifest.get("root", "")),
        items=items,
        splits=splits,
        metadata=dict(metadata),
        license=manifest.get("license"),
        source_url=manifest.get("source_url"),
        notes=manifest.get("notes"),
    )


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return normalize_dataset_manifest(data)


def dataset_manifest_to_dict(manifest: DatasetManifest) -> dict[str, Any]:
    return asdict(manifest)


def _is_relative_child_path(path_text: str) -> bool:
    path = Path(path_text)
    if path.is_absolute():
        return False
    return ".." not in path.parts


def _recommended_metadata_warnings(item: DatasetItem, domain: str) -> list[str]:
    warnings: list[str] = []
    if domain == "image":
        if item.width is None or item.height is None:
            warnings.append(f"item_recommended_image_dimensions_missing:{item.item_id}")
    elif domain == "audio":
        for field_name in ("duration_s", "sample_rate", "channels"):
            if getattr(item, field_name) is None:
                warnings.append(
                    f"item_recommended_audio_{field_name}_missing:{item.item_id}"
                )
    elif domain == "video":
        for field_name in ("width", "height", "fps", "num_frames"):
            if getattr(item, field_name) is None:
                warnings.append(
                    f"item_recommended_video_{field_name}_missing:{item.item_id}"
                )
    return warnings


def _item_metadata_summary(manifest: DatasetManifest) -> dict[str, Any]:
    item_keys: set[str] = set()
    items_with_metadata = 0
    for item in manifest.items:
        if item.metadata:
            items_with_metadata += 1
            item_keys.update(str(key) for key in item.metadata)

    return {
        "dataset_metadata_keys": sorted(str(key) for key in manifest.metadata),
        "item_metadata_keys": sorted(item_keys),
        "items_with_metadata": items_with_metadata,
        "items_without_metadata": len(manifest.items) - items_with_metadata,
    }


def validate_dataset_manifest(
    manifest: DatasetManifest | Mapping[str, Any],
    *,
    check_files: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    try:
        normalized = normalize_dataset_manifest(manifest)
    except Exception as exc:
        return {
            "valid": False,
            "errors": [f"manifest_coercion_failed:{type(exc).__name__}:{exc}"],
            "warnings": warnings,
            "num_items": 0,
            "num_splits": 0,
            "domain": None,
            "root": None,
            "missing_files": 0,
            "item_metadata_summary": {},
            "normalized_manifest": None,
        }

    if not normalized.schema_version.strip():
        errors.append("missing_schema_version")
    if not DATASET_ID_RE.match(normalized.dataset_id):
        errors.append(f"invalid_dataset_id:{normalized.dataset_id}")
    if normalized.domain not in VALID_DOMAINS:
        errors.append(f"invalid_domain:{normalized.domain}")
    if not normalized.root.strip():
        errors.append("missing_root")

    seen_item_ids: set[str] = set()
    duplicate_item_ids: set[str] = set()
    missing_files = 0
    root_path = Path(normalized.root)

    if check_files and normalized.root.strip() and not root_path.exists():
        errors.append(f"missing_root_path:{normalized.root}")

    for index, item in enumerate(normalized.items):
        if not item.item_id.strip():
            errors.append(f"missing_item_id:{index}")
        elif item.item_id in seen_item_ids:
            duplicate_item_ids.add(item.item_id)
        else:
            seen_item_ids.add(item.item_id)

        if not item.path.strip():
            errors.append(f"missing_item_path:{item.item_id or index}")
        elif not _is_relative_child_path(item.path):
            errors.append(f"item_path_not_relative_to_root:{item.item_id}:{item.path}")
        elif check_files:
            item_path = root_path / item.path
            if not item_path.exists():
                missing_files += 1
                errors.append(f"missing_item_file:{item.item_id}:{item.path}")

        warnings.extend(_recommended_metadata_warnings(item, normalized.domain))

    for item_id in sorted(duplicate_item_ids):
        errors.append(f"duplicate_item_id:{item_id}")

    item_ids = {item.item_id for item in normalized.items}
    for split_name, split_item_ids in normalized.splits.items():
        if not split_name.strip():
            errors.append("empty_split_name")
        for item_id in split_item_ids:
            if item_id not in item_ids:
                errors.append(f"split_item_id_not_found:{split_name}:{item_id}")

    if not normalized.metadata:
        warnings.append("dataset_metadata_empty")
    if normalized.license is None:
        warnings.append("license_missing")
    if normalized.source_url is None:
        warnings.append("source_url_missing")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "num_items": len(normalized.items),
        "num_splits": len(normalized.splits),
        "domain": normalized.domain,
        "root": normalized.root,
        "missing_files": missing_files,
        "item_metadata_summary": _item_metadata_summary(normalized),
        "normalized_manifest": dataset_manifest_to_dict(normalized),
    }


def manifest_to_item_table(
    manifest: DatasetManifest | Mapping[str, Any],
) -> list[dict[str, Any]]:
    normalized = normalize_dataset_manifest(manifest)
    rows: list[dict[str, Any]] = []
    for item in normalized.items:
        row = {
            "dataset_id": normalized.dataset_id,
            "domain": normalized.domain,
            "root": normalized.root,
            "item_id": item.item_id,
            "path": item.path,
            "width": item.width,
            "height": item.height,
            "pixels": item.pixels,
            "duration_s": item.duration_s,
            "sample_rate": item.sample_rate,
            "channels": item.channels,
            "fps": item.fps,
            "num_frames": item.num_frames,
            "metadata_json": json.dumps(item.metadata, sort_keys=True),
        }
        for key, value in item.metadata.items():
            row[f"metadata_{key}"] = value
        rows.append(row)
    return rows


def validate_manifest_against_domain_spec(
    manifest: DatasetManifest | Mapping[str, Any],
    domain_spec: DomainSpec | Mapping[str, Any] | str,
) -> dict[str, Any]:
    normalized = normalize_dataset_manifest(manifest)
    spec = resolve_domain_spec(domain_spec) if isinstance(domain_spec, str) else domain_spec
    spec_domain = spec.domain if isinstance(spec, DomainSpec) else str(spec.get("domain", ""))

    errors: list[str] = []
    warnings: list[str] = []
    if normalized.domain != spec_domain:
        errors.append(
            f"domain_mismatch:manifest={normalized.domain}:domain_spec={spec_domain}"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "num_items": len(normalized.items),
        "num_splits": len(normalized.splits),
        "domain": normalized.domain,
        "root": normalized.root,
        "missing_files": 0,
        "item_metadata_summary": _item_metadata_summary(normalized),
        "normalized_manifest": dataset_manifest_to_dict(normalized),
        "domain_spec_domain": spec_domain,
    }


def _write_item_table_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                columns.append(key)
                seen.add(key)

    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary_report(manifest: DatasetManifest | Mapping[str, Any]) -> dict[str, Any]:
    normalized = normalize_dataset_manifest(manifest)
    return {
        "valid": True,
        "errors": [],
        "warnings": [],
        "num_items": len(normalized.items),
        "num_splits": len(normalized.splits),
        "domain": normalized.domain,
        "root": normalized.root,
        "missing_files": 0,
        "item_metadata_summary": _item_metadata_summary(normalized),
    }


def build_dataset_manifest_template(domain: str) -> dict[str, Any]:
    domain = domain.strip().lower()
    if domain == "image":
        return {
            "schema_version": "dataset_manifest_v1",
            "dataset_id": "my_images",
            "display_name": "My Image Dataset",
            "domain": "image",
            "root": "datasets/my_images",
            "items": [
                {
                    "item_id": "img001",
                    "path": "img001.png",
                    "width": 768,
                    "height": 512,
                    "metadata": {"source": "template"},
                }
            ],
            "splits": {"all": ["img001"], "test": ["img001"]},
            "metadata": {"description": "Template image dataset manifest."},
            "license": "TODO",
            "source_url": "TODO",
            "notes": "Paths are examples; validate with --check-files when files exist.",
        }
    if domain == "audio":
        return {
            "schema_version": "dataset_manifest_v1",
            "dataset_id": "my_audio",
            "display_name": "My Audio Dataset",
            "domain": "audio",
            "root": "datasets/my_audio",
            "items": [
                {
                    "item_id": "aud001",
                    "path": "aud001.wav",
                    "duration_s": 3.5,
                    "sample_rate": 48000,
                    "channels": 2,
                    "metadata": {"source": "template"},
                }
            ],
            "splits": {"all": ["aud001"], "test": ["aud001"]},
            "metadata": {"description": "Template audio dataset manifest."},
            "license": "TODO",
            "source_url": "TODO",
            "notes": "Paths are examples; validate with --check-files when files exist.",
        }
    if domain == "video":
        return {
            "schema_version": "dataset_manifest_v1",
            "dataset_id": "my_video",
            "display_name": "My Video Dataset",
            "domain": "video",
            "root": "datasets/my_video",
            "items": [
                {
                    "item_id": "vid001",
                    "path": "vid001.y4m",
                    "width": 1920,
                    "height": 1080,
                    "fps": 30.0,
                    "num_frames": 120,
                    "metadata": {"source": "template"},
                }
            ],
            "splits": {"all": ["vid001"], "test": ["vid001"]},
            "metadata": {"description": "Template video dataset manifest."},
            "license": "TODO",
            "source_url": "TODO",
            "notes": "Paths are examples; validate with --check-files when files exist.",
        }
    raise ValueError(f"Unsupported dataset template domain: {domain}")


def _json_ready_report(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate dataset manifests and export item tables."
    )
    parser.add_argument("--manifest", default=None, help="Dataset manifest JSON path.")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--to-csv", default=None, help="Write an item table CSV.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for --new-template.",
    )
    parser.add_argument(
        "--new-template",
        choices=["image", "audio", "video"],
        default=None,
        help="Write a starter DatasetManifest JSON template.",
    )
    parser.add_argument("--print-summary", action="store_true")
    parser.add_argument("--domain-spec", default=None, help="DomainSpec builtin or JSON path.")
    parser.add_argument("--validate-domain", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.new_template is not None:
        if args.out is None:
            parser.error("--new-template requires --out")
        template = build_dataset_manifest_template(args.new_template)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(template, indent=2), encoding="utf-8")
        report = validate_dataset_manifest(template)
        report["template_out"] = str(out_path)
        print(_json_ready_report(report))
        return report

    if args.manifest is None:
        parser.print_help()
        return {"valid": True, "errors": [], "warnings": []}

    manifest = load_dataset_manifest(args.manifest)

    if args.validate_domain:
        if args.domain_spec is None:
            parser.error("--validate-domain requires --domain-spec")
        report = validate_manifest_against_domain_spec(manifest, args.domain_spec)
        print(_json_ready_report(report))
        return report

    if args.to_csv:
        rows = manifest_to_item_table(manifest)
        _write_item_table_csv(rows, args.to_csv)
        report = validate_dataset_manifest(manifest, check_files=args.check_files)
        report["item_table_csv"] = str(args.to_csv)
        print(_json_ready_report(report))
        return report

    if args.print_summary:
        report = _summary_report(manifest)
        print(_json_ready_report(report))
        return report

    if args.validate:
        report = validate_dataset_manifest(manifest, check_files=args.check_files)
        print(_json_ready_report(report))
        return report

    parser.print_help()
    return {"valid": True, "errors": [], "warnings": []}


if __name__ == "__main__":
    main()
