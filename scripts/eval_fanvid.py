#!/usr/bin/env python3
"""FANVID evaluation — face + license plate accuracy against ground truth.

Usage:
  python scripts/eval_fanvid.py [--num-videos 10] [--seed 42] [--model-size medium]

Pipeline:
  1. Download FANVID metadata from HuggingFace
  2. Download HR mugshots → build face embedding gallery via insightface
  3. Pick N random test clips (face + license plate)
  4. Download each clip from YouTube (yt-dlp, trimmed)
  5. Run full Sentinel Vision pipeline on each
  6. Compare pipeline output to ground-truth annotations
  7. Print per-clip and aggregate metrics
"""

import argparse
import csv
import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analytics.face_recognition import FaceRecognizer
from src.pipeline import analyze_video

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("eval_fanvid")
log.setLevel(logging.INFO)

HF_REPO = "kv1388/FANVID-Face_and_License_Plate_Recognition_in_Low-Resolution_Videos"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/data"
METADATA_FILES = [
    "Celebrity_mugshot.csv",
    "dataset_celebs.csv",
    "dataset_lp.csv",
    "celebrity_annotations_LR.csv",
    "license_plate_annotations_LR.csv",
]


def download_metadata(data_dir: Path):
    """Download all FANVID CSV files from HuggingFace."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for fname in METADATA_FILES:
        path = data_dir / fname
        if path.exists() and path.stat().st_size > 0:
            log.info("  [skip] %s exists", fname)
            continue
        url = f"{HF_BASE}/{fname}"
        log.info("  Downloading %s ...", fname)
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    log.info("Metadata downloaded to %s", data_dir)


def load_mugshots(mugshot_csv: Path) -> list[tuple[str, str]]:
    """Return [(name, image_url)] from Celebrity_mugshot.csv."""
    rows = []
    with open(mugshot_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["CelebName"].strip()
            url = row["RefImage"].strip().strip('"').strip("'")
            if name and url:
                rows.append((name, url))
    log.info("Loaded %d mugshot references", len(rows))
    return rows


def download_mugshot(url: str, timeout: int = 30) -> np.ndarray | None:
    """Download image from URL and return as RGB numpy array."""
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        img_bytes = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        log.warning("  Failed to download %s: %s", url[:60], e)
        return None


def build_face_gallery(
    fr: FaceRecognizer, mugshot_csv: Path,
    max_faces: int | None = None,
) -> int:
    """Populate FaceRecognizer gallery from HR mugshots.

    Returns number of successfully added faces.
    """
    mugshots = load_mugshots(mugshot_csv)
    if max_faces:
        mugshots = mugshots[:max_faces]

    success = 0
    for name, url in tqdm(mugshots, desc="Building face gallery"):
        if name in fr.gallery.known_names():
            continue
        img = download_mugshot(url)
        if img is None:
            continue
        if fr.add_known_face(name, img):
            success += 1
            log.info("  Added '%s' to gallery (%d/%d)", name, success, len(mugshots))
        else:
            log.warning("  No face detected in mugshot for '%s'", name)

    log.info(
        "Gallery built: %d/%d faces added (total: %d known)",
        success, len(mugshots), len(fr.gallery.known_names()),
    )
    return success


def load_dataset(dataset_csv: Path) -> list[dict]:
    """Load clip metadata from dataset_celebs.csv or dataset_lp.csv."""
    clips = []
    with open(dataset_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_id = row.get("Clip ID", "").strip()
            name = row.get("Name", "").strip()
            video_id = row.get("Video ID", "").strip()
            you_tube_url = row.get("You_Tube_URL", "").strip()
            try:
                start_s = float(row.get("Start time (s)", "0"))
                end_s = float(row.get("End time (s)", "0"))
            except ValueError:
                continue
            fps = float(row.get("FPS", "25"))
            split = row.get("Split", "train").strip().lower()
            clips.append({
                "clip_id": clip_id,
                "name": name,
                "video_id": video_id,
                "url": you_tube_url or f"https://www.youtube.com/watch?v={video_id}",
                "start_s": start_s,
                "end_s": end_s,
                "fps": fps,
                "split": split,
            })
    log.info("Loaded %d clips from %s", len(clips), dataset_csv.name)
    return clips


def load_annotations(annot_csv: Path) -> dict[str, list[dict]]:
    """Load LR annotations, group by Clip ID.

    Returns {clip_id_str: [annotation_dict, ...]}
    """
    groups: dict[str, list[dict]] = {}
    with open(annot_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_id = str(int(float(row["Clip ID"])))
            groups.setdefault(clip_id, []).append({
                "frame_no": int(float(row["FrameNo"])),
                "box_left": float(row["BoxLeft"]),
                "box_right": float(row["BoxRight"]),
                "box_top": float(row["BoxTop"]),
                "box_bottom": float(row["BoxBottom"]),
                "identity": row.get("IdentityOrText", "").strip(),
            })
    log.info("Loaded annotations for %d clips from %s", len(groups), annot_csv.name)
    return groups


def download_clip(clip: dict, output_dir: Path) -> Path | None:
    """Download trimmed YouTube clip via yt-dlp.

    Returns path to downloaded MP4, or None on failure.
    """
    out_name = f"clip_{clip['clip_id']}_{clip['video_id']}.mp4"
    out_path = output_dir / out_name
    if out_path.exists() and out_path.stat().st_size > 1000:
        log.info("  [skip] %s exists", out_name)
        return out_path

    # Use yt-dlp with download-time slicing
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--merge-output-format", "mp4",
        "--external-downloader", "ffmpeg",
        "--external-downloader-args",
        f"ffmpeg_i:-ss {clip['start_s']} -to {clip['end_s']}",
        "-o", str(out_path),
        clip["url"],
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        log.warning("  Timeout downloading %s", clip["url"])
        return None

    if out_path.exists() and out_path.stat().st_size > 1000:
        return out_path
    # Try alternative: download full then trim locally
    log.info("  Retrying: download full + trim locally for %s", clip["url"])
    full_path = output_dir / f"full_{clip['video_id']}.mp4"
    dl_cmd = [
        "yt-dlp", "--quiet", "--no-warnings",
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--merge-output-format", "mp4",
        "-o", str(full_path),
        clip["url"],
    ]
    try:
        subprocess.run(dl_cmd, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None
    if not full_path.exists():
        return None

    duration = clip["end_s"] - clip["start_s"]
    trim_cmd = [
        "ffmpeg", "-y",
        "-ss", str(clip["start_s"]),
        "-i", str(full_path),
        "-to", str(duration),
        "-c:v", "libx264", "-preset", "fast",
        "-an",  # no audio
        str(out_path),
    ]
    subprocess.run(trim_cmd, capture_output=True, timeout=120)
    full_path.unlink(missing_ok=True)

    if out_path.exists() and out_path.stat().st_size > 1000:
        return out_path
    return None


def run_pipeline_eval(
    video_path: Path,
    clip: dict,
    annotations: list[dict] | None,
    device: str,
    model_size: str,
    output_root: Path,
) -> dict:
    """Run Sentinel Vision pipeline on one clip and compare to ground truth.

    Returns dict with evaluation results.
    """
    clip_label = f"{clip['name']} [{clip['clip_id']}]"
    out_dir = output_root / f"clip_{clip['clip_id']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = analyze_video(
            video_path=str(video_path),
            output_dir=str(out_dir),
            model_family="yolo11",
            model_size=model_size,
            conf_threshold=0.4,
            device=device,
            max_frames=None,
            use_reid=True,
            reid_model="x1_0",
            track_thresh=0.4,
            match_thresh=0.7,
            track_low_thresh=0.1,
            track_buffer=450,
            capture_evidence=False,
            filter_stationary_objects=False,
            min_move_distance=5.0,
        )
    except Exception as e:
        log.error("Pipeline failed on %s: %s", clip_label, e)
        return {"clip": clip_label, "ground_truth": clip["name"], "error": str(e)}

    # ── Face evaluation ──
    identities = result.get("identities", [])
    recognized_names = {id_["name"] for id_ in identities}
    gt = clip["name"]

    face_correct = gt in recognized_names
    face_wrong = bool(recognized_names) and not face_correct
    face_miss = not recognized_names
    num_faces = len(identities)

    return {
        "clip": clip_label,
        "ground_truth": gt,
        "face_correct": face_correct,
        "face_wrong": face_wrong,
        "face_miss": face_miss,
        "num_faces": num_faces,
        "recognized_names": list(recognized_names),
        "video_frames": result.get("total_frames_processed", 0),
        "output_dir": str(out_dir),
    }


def run_lp_eval(
    video_path: Path,
    clip: dict,
    annotations: list[dict] | None,
    device: str,
    model_size: str,
    output_root: Path,
) -> dict:
    """Run pipeline on LP clip, compare detected plate to ground truth.

    Uses edit distance (Levenshtein) for plate text comparison.
    """
    clip_label = f"LP-{clip['name']} [{clip['clip_id']}]"
    out_dir = output_root / f"lp_clip_{clip['clip_id']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = analyze_video(
            video_path=str(video_path),
            output_dir=str(out_dir),
            model_family="yolo11",
            model_size=model_size,
            conf_threshold=0.4,
            device=device,
            max_frames=None,
            use_reid=True,
            capture_evidence=False,
            filter_stationary_objects=False,
            min_move_distance=5.0,
        )
    except Exception as e:
        log.error("Pipeline failed on %s: %s", clip_label, e)
        return {"clip": clip_label, "ground_truth": clip["name"], "error": str(e)}

    # ── LP evaluation ──
    gt_plate = clip["name"].strip().upper()
    detected_plates = []
    for v in result.get("vehicle_list", []):
        plate = v.get("plate")
        if plate:
            detected_plates.append(plate.strip().upper())

    # Edit distance
    if detected_plates:
        from difflib import SequenceMatcher
        best_plate = detected_plates[0]
        best_dist = 1.0 - SequenceMatcher(None, best_plate, gt_plate).ratio()
        for p in detected_plates:
            dist = 1.0 - SequenceMatcher(None, p, gt_plate).ratio()
            if dist < best_dist:
                best_dist = dist
                best_plate = p
    else:
        best_plate = ""
        best_dist = 1.0

    correct = best_plate == gt_plate

    return {
        "clip": clip_label,
        "ground_truth": gt_plate,
        "detected_plates": detected_plates,
        "best_match": best_plate,
        "edit_distance": best_dist,
        "correct": correct,
        "video_frames": result.get("total_frames_processed", 0),
        "output_dir": str(out_dir),
    }


_GLOBAL_T0 = time.time()

def print_report(face_results: list[dict], lp_results: list[dict]):
    """Print formatted evaluation report."""
    total_time = time.time() - _GLOBAL_T0

    print("\n" + "=" * 70)
    print("  FANVID EVALUATION REPORT")
    print("=" * 70)

    # ── Face results ──
    print(f"\n{'─' * 70}")
    print(f"  FACE RECOGNITION ({len(face_results)} clips)")
    print(f"{'─' * 70}")
    if face_results:
        correct = sum(1 for r in face_results if r.get("face_correct"))
        wrong = sum(1 for r in face_results if r.get("face_wrong"))
        missed = sum(1 for r in face_results if r.get("face_miss"))
        errors = sum(1 for r in face_results if r.get("error"))

        print(f"  {'Clip':<40s} {'GT':>20s} {'Result':>10s}")
        print(f"  {'─' * 70}")
        for r in face_results:
            if r.get("error"):
                status = "ERROR"
            elif r.get("face_correct"):
                status = "✓ CORRECT"
            elif r.get("face_wrong"):
                status = "✗ WRONG"
            else:
                status = "○ MISS"

            names = r.get("recognized_names", [])
            names_str = ", ".join(names[:3]) if names else "—"
            label = f"{r['clip'][:38]}"
            print(f"  {label:<40s} {r['ground_truth'][:20]:>20s} {status:>10s}")
            if names:
                print(f"  {'':40s} recognized: {names_str[:35]}")

        print(f"\n  Summary:")
        print(f"    Total clips:       {len(face_results)}")
        print(f"    Correct:           {correct} ({correct/len(face_results)*100:.0f}%)")
        print(f"    Wrong identity:    {wrong}")
        print(f"    Not recognized:    {missed}")
        print(f"    Pipeline errors:   {errors}")
        print(f"    Accuracy (C/T):    {correct}/{len(face_results)} = {correct/len(face_results)*100:.1f}%")
    else:
        print("  No face clips evaluated.")

    # ── License Plate results ──
    print(f"\n{'─' * 70}")
    print(f"  LICENSE PLATE RECOGNITION ({len(lp_results)} clips)")
    print(f"{'─' * 70}")
    if lp_results:
        correct = sum(1 for r in lp_results if r.get("correct"))
        errors = sum(1 for r in lp_results if r.get("error"))

        print(f"  {'Clip':<40s} {'GT Plate':>15s} {'Detected':>15s} {'Dist':>6s}")
        print(f"  {'─' * 70}")
        for r in lp_results:
            if r.get("error"):
                print(f"  {r['clip'][:38]:<40s} {'ERROR':>36s}")
            else:
                det = r.get("best_match", "—")
                label = f"{r['clip'][:38]}"
                status = "✓" if r.get("correct") else "✗"
                print(f"  {label:<40s} {r['ground_truth'][:15]:>15s} {det[:15]:>15s} {r['edit_distance']:>5.2f} {status}")

        print(f"\n  Summary:")
        print(f"    Total clips:       {len(lp_results)}")
        print(f"    Exact matches:     {correct} ({correct/len(lp_results)*100:.0f}%)")
        print(f"    Pipeline errors:   {errors}")
        avg_dist = np.mean([r.get("edit_distance", 1.0) for r in lp_results if not r.get("error")])
        print(f"    Avg edit distance: {avg_dist:.3f}")
    else:
        print("  No license plate clips evaluated.")

    # ── Overall ──
    print(f"\n{'─' * 70}")
    print(f"  SYSTEM OVERVIEW")
    print(f"{'─' * 70}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    total_frames = sum(r.get("video_frames", 0) for r in face_results + lp_results if not r.get("error"))
    print(f"  Total frames processed: {total_frames}")
    print(f"\n{'=' * 70}")


# ── main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FANVID evaluation")
    parser.add_argument("--num-videos", type=int, default=10,
                        help="Number of random videos to test")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--model-size", default="nano",
                        choices=["nano", "small", "medium", "large", "xlarge"],
                        help="YOLO model size")
    parser.add_argument("--data-dir", default="data/FANVID",
                        help="Directory for FANVID data")
    parser.add_argument("--output-dir", default="outputs/fanvid_eval",
                        help="Output directory for eval results")
    parser.add_argument("--face-only", action="store_true",
                        help="Only evaluate face clips (skip LP)")
    parser.add_argument("--lp-only", action="store_true",
                        help="Only evaluate LP clips (skip face)")
    parser.add_argument("--max-gallery", type=int, default=None,
                        help="Max faces to add to gallery (for quick testing)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip video downloading (use existing files)")
    args = parser.parse_args()

    global _GLOBAL_T0
    _GLOBAL_T0 = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model:  YOLO11{args.model_size}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Download metadata ──
    print("\n[1/5] Downloading FANVID metadata...")
    download_metadata(data_dir)

    mugshot_csv = data_dir / "Celebrity_mugshot.csv"
    celebs_csv = data_dir / "dataset_celebs.csv"
    lp_csv = data_dir / "dataset_lp.csv"
    face_annot_csv = data_dir / "celebrity_annotations_LR.csv"
    lp_annot_csv = data_dir / "license_plate_annotations_LR.csv"

    # ── Step 2: Build face gallery ──
    print("\n[2/5] Building face embedding gallery from HR mugshots...")
    fr = FaceRecognizer(
        gallery_path="face_gallery.json",
        device=device,
        capture_unknowns=False,
    )
    if not fr.available:
        log.warning("insightface not available — face evaluation will be limited")
    else:
        n_added = build_face_gallery(fr, mugshot_csv, max_faces=args.max_gallery)
        if n_added == 0:
            print("  Warning: no faces added to gallery. Face evaluation will likely fail.")
        else:
            print(f"  Gallery ready: {len(fr.gallery.known_names())} known faces")

    # ── Step 3: Select test clips ──
    print(f"\n[3/5] Selecting {args.num_videos} random test videos...")
    rng = random.Random(args.seed)

    face_clips = load_dataset(celebs_csv)
    test_clips = [c for c in face_clips if c["split"] == "test"]
    if not test_clips:
        test_clips = face_clips  # fallback to all splits
    if len(test_clips) > args.num_videos:
        test_clips = rng.sample(test_clips, args.num_videos)
    else:
        test_clips = test_clips[:]

    lp_data = load_dataset(lp_csv)
    lp_test = [c for c in lp_data if c["split"] == "test" or True]
    if len(lp_test) > max(2, args.num_videos // 3):
        lp_test = rng.sample(lp_test, max(2, args.num_videos // 3))

    print(f"  Selected {len(test_clips)} face clips, {len(lp_test)} LP clips")
    if args.face_only:
        lp_test = []
    if args.lp_only:
        test_clips = []

    # ── Step 4: Download and evaluate ──
    print("\n[4/5] Downloading and analyzing videos...")
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Face annotations lookup
    face_annot = load_annotations(face_annot_csv)
    lp_annot = load_annotations(lp_annot_csv)

    face_results = []
    for clip in tqdm(test_clips, desc="Face clips"):
        clip_annot = face_annot.get(clip["clip_id"])
        if not args.skip_download:
            vpath = download_clip(clip, videos_dir)
            if vpath is None:
                log.warning("  Skipping %s (download failed)", clip["name"])
                continue
        else:
            vpath = videos_dir / f"clip_{clip['clip_id']}_{clip['video_id']}.mp4"
            if not vpath.exists():
                log.warning("  Skipping %s (file not found)", clip["name"])
                continue

        result = run_pipeline_eval(
            vpath, clip, clip_annot, device,
            args.model_size, output_dir,
        )
        face_results.append(result)

        # Print per-clip result inline
        if result.get("face_correct"):
            symbol = "✓"
        elif result.get("face_wrong"):
            symbol = "✗"
        elif result.get("error"):
            symbol = "!"
        else:
            symbol = "○"
        print(f"  {symbol} {result['clip'][:50]} → {result['ground_truth'][:20]}")

    lp_results = []
    for clip in tqdm(lp_test, desc="LP clips"):
        clip_annot = lp_annot.get(clip["clip_id"])
        if not args.skip_download:
            vpath = download_clip(clip, videos_dir)
            if vpath is None:
                continue
        else:
            vpath = videos_dir / f"clip_{clip['clip_id']}_{clip['video_id']}.mp4"
            if not vpath.exists():
                continue

        result = run_lp_eval(
            vpath, clip, clip_annot, device,
            args.model_size, output_dir,
        )
        lp_results.append(result)

        symbol = "✓" if result.get("correct") else ("!" if result.get("error") else "✗")
        det = result.get("best_match", "—")[:15]
        print(f"  {symbol} {result['clip'][:40]} GT={result['ground_truth'][:12]} DET={det}")

    # ── Step 5: Report ──
    print("\n[5/5] Generating report...")
    print_report(face_results, lp_results)

    # Save results to JSON
    report_path = output_dir / "eval_results.json"
    report = {
        "model_size": args.model_size,
        "device": device,
        "num_face_clips": len(face_results),
        "num_lp_clips": len(lp_results),
        "face_accuracy": (
            sum(1 for r in face_results if r.get("face_correct")) / len(face_results)
            if face_results else 0
        ),
        "lp_accuracy": (
            sum(1 for r in lp_results if r.get("correct")) / len(lp_results)
            if lp_results else 0
        ),
        "face_results": [
            {k: v for k, v in r.items() if k != "recognized_names" or len(v) <= 5}
            for r in face_results
        ],
        "lp_results": [
            {k: v for k, v in r.items() if k != "detected_plates" or len(v) <= 5}
            for r in lp_results
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nFull results: {report_path}")


if __name__ == "__main__":
    main()
