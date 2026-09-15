import io
import os
import json
import time
import tarfile
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import webdataset as wds

assert "partition" in globals(), "Run the partition preflight cell first."

REPO_ID = "earthflow/GAMUS"
REVISION = "a3c0e2511f06d909612406f436cf8abb4da805f5"
PLAN_ID = "092af5247f89bec6"
PARTITION_ID = 2

CROP_WINDOWS = [
    {"tile_id": "nw", "x": 0,   "y": 0,   "width": 518, "height": 518},
    {"tile_id": "ne", "x": 506, "y": 0,   "width": 518, "height": 518},
    {"tile_id": "sw", "x": 0,   "y": 506, "width": 518, "height": 518},
    {"tile_id": "se", "x": 506, "y": 506, "width": 518, "height": 518},
]

HEIGHT_FILL_VALUE = np.float32(-5.0)
CLASS_IGNORE_VALUE = np.uint8(255)
MAX_SHARD_BYTES = 512 * 1024**2
OUTPUT_ABORT_BYTES = 14 * 1024**3
OUTPUT_BUDGET_BYTES = int(14.5 * 1024**3)

BUILD_ROOT = Path("/kaggle/working/gamus-pretiles-02")
REPORT_PATH = BUILD_ROOT / "partition-report.json"
PROGRESS_PATH = BUILD_ROOT / "progress.json"

if BUILD_ROOT.exists():
    raise RuntimeError(
        f"{BUILD_ROOT} already exists. Do not overwrite a previous build; "
        "inspect it first and use a fresh output path only if instructed."
    )

records = sorted(
    partition["samples"],
    key=lambda item: (
        item["split"],
        item.get("city") or item["sample_id"].split("_", 1)[0],
        item["sample_id"],
    ),
)
expected_sources = len(records)
expected_pretiles = expected_sources * len(CROP_WINDOWS)

if expected_sources != 1454 or expected_pretiles != 5816:
    raise RuntimeError(
        f"Unexpected partition contents: {expected_sources} sources / "
        f"{expected_pretiles} pretiles."
    )

def city_of(record):
    return record.get("city") or record["sample_id"].split("_", 1)[0]

def png_bytes(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()

def npy_bytes(array):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()

def json_bytes(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

def tar_storage_bytes(payload):
    return 512 + ((len(payload) + 511) // 512) * 512

def add_bytes(tar, member_name, payload):
    info = tarfile.TarInfo(member_name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    tar.addfile(info, io.BytesIO(payload))

class SplitTarWriter:
    def __init__(self, split):
        self.split = split
        self.directory = BUILD_ROOT / split
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shard_index = 0
        self.tar = None
        self.partial_path = None
        self.final_path = None
        self.active_records = 0
        self.active_estimated_bytes = 0
        self.finalized_bytes = 0
        self.shards = []
        self._open_new_shard()

    def _open_new_shard(self):
        self.partial_path = self.directory / f"{self.split}-{self.shard_index:06d}.partial.tar"
        self.final_path = self.directory / f"{self.split}-{self.shard_index:06d}.tar"
        self.tar = tarfile.open(self.partial_path, mode="w")
        self.active_records = 0
        self.active_estimated_bytes = 1024  # TAR end blocks
        self.shard_index += 1

    def write_record(self, members):
        record_bytes = sum(tar_storage_bytes(payload) for payload in members.values())

        if self.active_records > 0 and self.active_estimated_bytes + record_bytes > MAX_SHARD_BYTES:
            self.close_active_shard()
            self._open_new_shard()

        for member_name, payload in members.items():
            add_bytes(self.tar, member_name, payload)

        self.active_records += 1
        self.active_estimated_bytes += record_bytes

    def close_active_shard(self):
        if self.tar is None:
            return

        self.tar.close()

        if self.active_records == 0:
            self.partial_path.unlink(missing_ok=True)
        else:
            os.replace(self.partial_path, self.final_path)
            actual_bytes = self.final_path.stat().st_size
            self.finalized_bytes += actual_bytes
            self.shards.append(
                {
                    "split": self.split,
                    "path": str(self.final_path),
                    "record_count": self.active_records,
                    "tar_size_bytes": actual_bytes,
                }
            )

        self.tar = None

    def abort_active_shard(self):
        if self.tar is not None:
            self.tar.close()
            self.tar = None
        # A .partial.tar is intentionally left behind after failure and is never a valid shard.

    @property
    def bytes_so_far(self):
        return self.finalized_bytes + (self.active_estimated_bytes if self.tar is not None else 0)

BUILD_ROOT.mkdir(parents=True)
writers = {split: SplitTarWriter(split) for split in ("train", "val", "test")}

source_by_split = Counter()
source_by_city = Counter()
label_stats = {
    "finite_pixels": 0,
    "nonfinite_pixels": 0,
    "exact_minus_5_pixels": 0,
    "valid_pixels": 0,
    "negative_nonfill_pixels": 0,
    "finite_height_min": float("inf"),
    "finite_height_max": float("-inf"),
}

build_succeeded = False
start_time = time.perf_counter()

try:
    for source_number, record in enumerate(records, start=1):
        split = record["split"]
        city = city_of(record)

        with tempfile.TemporaryDirectory(prefix="gamus-build-", dir="/tmp") as temp_dir:
            def download_source(filename):
                return hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    revision=REVISION,
                    filename=filename,
                    local_dir=temp_dir,
                )

            image_file = download_source(record["image_path"])
            height_file = download_source(record["height_path"])
            class_file = download_source(record["class_path"])

            with h5py.File(image_file, "r") as handle:
                rgb = np.asarray(handle["image"])
            with h5py.File(height_file, "r") as handle:
                height_map = np.asarray(handle["image"], dtype=np.float32)
            with h5py.File(class_file, "r") as handle:
                classes_raw = np.asarray(handle["image"])

            if rgb.shape != (1024, 1024, 3) or rgb.dtype != np.uint8:
                raise ValueError(f"Unexpected RGB schema for {record['sample_id']}: {rgb.shape}, {rgb.dtype}")
            if height_map.shape != (1024, 1024):
                raise ValueError(f"Unexpected height schema for {record['sample_id']}: {height_map.shape}")
            if classes_raw.shape != (1024, 1024) or not np.isfinite(classes_raw).all():
                raise ValueError(f"Unexpected class schema for {record['sample_id']}")

            rounded_classes = np.rint(classes_raw)
            if not np.array_equal(classes_raw, rounded_classes):
                raise ValueError(f"Non-integral class values in {record['sample_id']}")
            if rounded_classes.min() < 0 or rounded_classes.max() > 6:
                raise ValueError(f"Class values outside 0..6 in {record['sample_id']}")

            classes = rounded_classes.astype(np.uint8)
            finite = np.isfinite(height_map)
            exact_minus_5 = height_map == HEIGHT_FILL_VALUE
            valid = finite & ~exact_minus_5

            label_stats["finite_pixels"] += int(finite.sum())
            label_stats["nonfinite_pixels"] += int((~finite).sum())
            label_stats["exact_minus_5_pixels"] += int(exact_minus_5.sum())
            label_stats["valid_pixels"] += int(valid.sum())
            label_stats["negative_nonfill_pixels"] += int(
                (finite & (height_map < 0) & ~exact_minus_5).sum()
            )

            if finite.any():
                label_stats["finite_height_min"] = min(
                    label_stats["finite_height_min"], float(height_map[finite].min())
                )
                label_stats["finite_height_max"] = max(
                    label_stats["finite_height_max"], float(height_map[finite].max())
                )

            for crop in CROP_WINDOWS:
                x, y = crop["x"], crop["y"]
                width, tile_height = crop["width"], crop["height"]

                rgb_tile = np.ascontiguousarray(rgb[y:y + tile_height, x:x + width, :])
                height_tile = np.ascontiguousarray(height_map[y:y + tile_height, x:x + width])
                class_tile = np.ascontiguousarray(classes[y:y + tile_height, x:x + width])
                valid_tile = np.ascontiguousarray(
                    valid[y:y + tile_height, x:x + width].astype(np.uint8) * 255
                )

                key = f"{split}--{record['sample_id']}--{crop['tile_id']}"
                metadata = {
                    "schema": "geoheight-wds-v1",
                    "plan_id": PLAN_ID,
                    "source_repo": REPO_ID,
                    "source_revision": REVISION,
                    "partition_id": PARTITION_ID,
                    "split": split,
                    "city": city,
                    "sample_id": record["sample_id"],
                    "tile_id": crop["tile_id"],
                    "crop": crop,
                    "height_valid_rule": "finite_and_not_exact_neg_5",
                    "invalid_pixel_count": int((valid_tile == 0).sum()),
                }

                writers[split].write_record(
                    {
                        f"{key}.rgb.png": png_bytes(rgb_tile),
                        f"{key}.height.npy": npy_bytes(height_tile),
                        f"{key}.class.png": png_bytes(class_tile),
                        f"{key}.valid.png": png_bytes(valid_tile),
                        f"{key}.json": json_bytes(metadata),
                    }
                )

        source_by_split[split] += 1
        source_by_city[city] += 1

        estimated_output_bytes = sum(writer.bytes_so_far for writer in writers.values())
        if estimated_output_bytes > OUTPUT_ABORT_BYTES:
            raise RuntimeError(
                "Safety stop: estimated output exceeded 14 GiB. "
                "Do not continue or publish this incomplete build."
            )

        if source_number % 10 == 0 or source_number == expected_sources:
            elapsed = time.perf_counter() - start_time
            eta_minutes = (elapsed / source_number) * (expected_sources - source_number) / 60
            progress = {
                "completed_sources": source_number,
                "total_sources": expected_sources,
                "completed_pretiles": source_number * len(CROP_WINDOWS),
                "estimated_output_gib_so_far": round(estimated_output_bytes / 1024**3, 3),
                "elapsed_minutes": round(elapsed / 60, 2),
                "estimated_remaining_minutes": round(eta_minutes, 1),
            }
            PROGRESS_PATH.write_text(json.dumps(progress, indent=2))
            print(json.dumps(progress))

    build_succeeded = True

finally:
    for writer in writers.values():
        if build_succeeded:
            writer.close_active_shard()
        else:
            writer.abort_active_shard()

if not build_succeeded:
    raise RuntimeError("Build stopped before completion; do not publish this output directory.")

shards = [shard for writer in writers.values() for shard in writer.shards]
total_records = sum(shard["record_count"] for shard in shards)
total_output_bytes = sum(shard["tar_size_bytes"] for shard in shards)

if total_records != expected_pretiles:
    raise RuntimeError(f"Expected {expected_pretiles} records, wrote {total_records}.")
if total_output_bytes > OUTPUT_BUDGET_BYTES:
    raise RuntimeError("Final output exceeds the 14.5 GiB publishing budget.")

# Header-level validation of every shard: five members per pre-tiled record.
for shard in shards:
    with tarfile.open(shard["path"], mode="r") as tar:
        member_count = sum(1 for item in tar.getmembers() if item.isfile())
    expected_members = shard["record_count"] * 5
    if member_count != expected_members:
        raise RuntimeError(
            f"{shard['path']} has {member_count} members; expected {expected_members}."
        )
    shard["tar_member_count"] = member_count

# Full WebDataset compatibility check for one finalized shard from every split.
wds_checks = {}
for split in ("train", "val", "test"):
    split_shard = next(shard for shard in shards if shard["split"] == split)
    observed_records = sum(
        1 for _ in wds.WebDataset(split_shard["path"], shardshuffle=False)
    )
    if observed_records != split_shard["record_count"]:
        raise RuntimeError(
            f"WebDataset check failed for {split}: "
            f"{observed_records} != {split_shard['record_count']}"
        )
    wds_checks[split] = {
        "checked_shard": split_shard["path"],
        "records": observed_records,
    }

partial_files = list(BUILD_ROOT.rglob("*.partial.tar"))
if partial_files:
    raise RuntimeError(f"Unexpected incomplete shards: {[str(path) for path in partial_files]}")

label_stats["finite_height_min"] = round(label_stats["finite_height_min"], 6)
label_stats["finite_height_max"] = round(label_stats["finite_height_max"], 6)

report = {
    "schema": "geoheight-wds-v1",
    "status": "partition build passed",
    "plan_id": PLAN_ID,
    "partition_id": PARTITION_ID,
    "source_repo": REPO_ID,
    "source_revision": REVISION,
    "source_tiles": expected_sources,
    "pre_tiled_records": total_records,
    "crop_windows": CROP_WINDOWS,
    "source_counts_by_split": dict(sorted(source_by_split.items())),
    "source_counts_by_city": dict(sorted(source_by_city.items())),
    "label_stats": label_stats,
    "height_valid_rule": "finite_and_not_exact_neg_5",
    "output_size_gib": round(total_output_bytes / 1024**3, 3),
    "output_budget_gib": 14.5,
    "shard_count": len(shards),
    "shards": shards,
    "webdataset_checks": wds_checks,
    "working_free_gib_after_build": round(
        shutil.disk_usage("/kaggle/working").free / 1024**3, 3
    ),
    "elapsed_minutes": round((time.perf_counter() - start_time) / 60, 2),
}

REPORT_PATH.write_text(json.dumps(report, indent=2))
PROGRESS_PATH.write_text(json.dumps(report, indent=2))

print(json.dumps(report, indent=2))