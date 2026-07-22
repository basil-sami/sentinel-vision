#!/usr/bin/env python3
"""FANVID evaluation — face + license plate accuracy.

Two evaluation modes:
  - "pipeline" (default): runs full Sentinel Vision pipeline (YOLO + tracker +
    face_recognition) on each clip; works best for standard-res surveillance.
  - "direct": runs insightface directly on each LR frame (bypasses YOLO/tracker);
    designed specifically for FANVID's 180×320 low-resolution clips.

Usage:
  python scripts/eval_fanvid.py --num-videos 10 --mode direct
  python scripts/eval_fanvid.py --num-videos 10 --mode pipeline  --min-face-size 20
  python scripts/eval_fanvid.py --num-videos 10 --mode both      # compare
"""

import argparse
import csv
import json
import logging
import random
import subprocess
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analytics.face_recognition import FaceRecognizer, FaceGallery
from src.pipeline import analyze_video

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
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


# ── helpers ──────────────────────────────────────────────────────────

def _transcode_av1(path: Path) -> Path | None:
    """Transcode AV1 → H.264 via ffmpeg. Returns new path or None."""
    out = path.with_suffix(".h264.mp4")
    cmd = ["ffmpeg", "-y", "-i", str(path),
           "-c:v", "libx264", "-preset", "fast",
           "-an", str(out)]
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    if r.returncode == 0 and out.exists() and out.stat().st_size > 1000:
        return out
    return None


def _has_frames(path: Path) -> bool:
    """Return True if video has at least 1 decodable frame."""
    cap = cv2.VideoCapture(str(path))
    ret, _ = cap.read()
    cap.release()
    return ret


# ── Step 1: Download metadata ──

def download_metadata(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    for fname in METADATA_FILES:
        path = data_dir / fname
        if path.exists() and path.stat().st_size > 0:
            continue
        r = requests.get(f"{HF_BASE}/{fname}", timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    log.info("Metadata → %s", data_dir)


# ── Step 2: Build face gallery ──

def load_mugshots(mugshot_csv: Path) -> list[tuple[str, str]]:
    rows = []
    with open(mugshot_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["CelebName"].strip()
            url = row["RefImage"].strip().strip('"').strip("'")
            if name and url:
                rows.append((name, url))
    return rows


def download_mugshot_with_retry(url: str, retries: int = 2) -> np.ndarray | None:
    """Download image; retry with different UA on 403/429."""
    uas = [
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "insightface/1.0",
    ]
    for ua in uas:
        try:
            r = requests.get(url, timeout=30,
                             headers={"User-Agent": ua})
            if r.status_code == 429:
                time.sleep(2)
                continue
            r.raise_for_status()
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            continue
    return None


def build_face_gallery(fr: FaceRecognizer, mugshot_csv: Path) -> int:
    mugshots = load_mugshots(mugshot_csv)
    already = set(fr.gallery.known_names())
    todo = [(n, u) for n, u in mugshots if n not in already]
    if not todo:
        log.info("Gallery already has %d faces, nothing to add", len(already))
        return len(already)
    success = 0
    for name, url in tqdm(todo, desc="Building face gallery"):
        img = download_mugshot_with_retry(url)
        if img is None:
            log.warning("  URL dead: %s", name)
            continue
        if fr.add_known_face(name, img):
            success += 1
            log.info("  Added: %s", name)
        else:
            log.warning("  No face in mugshot: %s", name)
    total = len(fr.gallery.known_names())
    log.info("Gallery: %d/%d faces added (total %d)", success, len(todo), total)
    return total


# ── Data loaders ──

def load_dataset(dataset_csv: Path) -> list[dict]:
    clips = []
    with open(dataset_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                start_s = float(row.get("Start time (s)", "0"))
                end_s = float(row.get("End time (s)", "0"))
            except ValueError:
                continue
            clips.append({
                "clip_id": row.get("Clip ID", "").strip(),
                "name": row.get("Name", "").strip(),
                "video_id": row.get("Video ID", "").strip(),
                "url": (row.get("You_Tube_URL", "").strip()
                        or f"https://www.youtube.com/watch?v={row.get('Video ID', '').strip()}"),
                "start_s": start_s,
                "end_s": end_s,
                "fps": float(row.get("FPS", "25")),
                "split": row.get("Split", "train").strip().lower(),
            })
    return clips


def load_annotations(annot_csv: Path) -> dict[str, list[dict]]:
    groups = {}
    with open(annot_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = str(int(float(row["Clip ID"])))
            groups.setdefault(cid, []).append({
                "frame_no": int(float(row["FrameNo"])),
                "box_left": float(row["BoxLeft"]),
                "box_right": float(row["BoxRight"]),
                "box_top": float(row["BoxTop"]),
                "box_bottom": float(row["BoxBottom"]),
                "identity": row.get("IdentityOrText", "").strip(),
            })
    return groups


# ── Video download ──

def download_clip(clip: dict, output_dir: Path) -> Path | None:
    out_name = f"clip_{clip['clip_id']}_{clip['video_id']}.mp4"
    out_path = output_dir / out_name
    if out_path.exists() and out_path.stat().st_size > 1000:
        return out_path

    # yt-dlp with download-time trimming
    cmd = [
        "yt-dlp", "--quiet", "--no-warnings",
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
        pass

    if out_path.exists() and out_path.stat().st_size > 1000:
        return out_path

    # Fallback: download full, trim locally
    full_path = output_dir / f"full_{clip['video_id']}.mp4"
    dl_cmd = [
        "yt-dlp", "--quiet", "--no-warnings",
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--merge-output-format", "mp4",
        "-o", str(full_path), clip["url"],
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
        "-i", str(full_path), "-to", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-an", str(out_path),
    ]
    subprocess.run(trim_cmd, capture_output=True, timeout=120)
    full_path.unlink(missing_ok=True)

    return out_path if (out_path.exists() and out_path.stat().st_size > 1000) else None


# ── Evaluation modes ──

def _ensure_good_video(path: Path) -> Path | None:
    """Transcode AV1 if needed; return usable path or None."""
    if not path.exists():
        return None
    if _has_frames(path):
        return path
    h264 = _transcode_av1(path)
    if h264 and _has_frames(h264):
        return h264
    return None


def eval_face_pipeline(
    video_path: Path, clip: dict, device: str,
    model_size: str, output_root: Path,
    min_face_size: int, face_interval: int,
) -> dict:
    """Run full pipeline and extract face identities."""
    clip_label = f"{clip['name']} [{clip['clip_id']}]"
    out_dir = output_root / f"pipe_{clip['clip_id']}"
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
            use_reid=False,
            track_thresh=0.4,
            match_thresh=0.7,
            track_low_thresh=0.1,
            track_buffer=120,
            capture_evidence=False,
            filter_stationary_objects=False,
            min_move_distance=5.0,
            min_face_size=min_face_size,
            face_interval=face_interval,
        )
    except Exception as e:
        return {"clip": clip_label, "gt": clip["name"], "error": str(e)[:100]}

    identities = result.get("identities", [])
    recognized = {id_["name"] for id_ in identities}
    gt = clip["name"]
    return {
        "clip": clip_label,
        "gt": gt,
        "correct": gt in recognized,
        "wrong": bool(recognized) and gt not in recognized,
        "miss": not recognized,
        "recognized": list(recognized)[:5],
        "num_identities": len(identities),
        "frames": result.get("total_frames_processed", 0),
    }


def eval_face_direct(
    video_path: Path, clip: dict, gallery: FaceGallery,
    min_face_size: int, match_threshold: float,
) -> dict:
    """Bypass YOLO/tracker — run insightface directly on every frame.

    Strategy: for each clip, extract ALL face embeddings across all frames,
    match each against the gallery, then pick the most frequent identity.
    """
    clip_label = f"{clip['name']} [{clip['clip_id']}]"
    gt = clip["name"]

    try:
        import insightface
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider" if torch.cuda.is_available()
                       else "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0 if torch.cuda.is_available() else -1,
                    det_size=(160, 160))
    except Exception as e:
        return {"clip": clip_label, "gt": gt, "error": f"insightface: {e}"[:100]}

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total == 0:
        cap.release()
        return {"clip": clip_label, "gt": gt, "error": "0 frames"}

    # Collect embeddings across frames
    all_embeddings = []
    frames_with_faces = 0
    for _ in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = app.get(rgb)
        if not faces:
            continue
        frames_with_faces += 1
        # Take the largest face per frame
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        if face.normed_embedding is not None:
            all_embeddings.append(face.normed_embedding)
    cap.release()

    if not all_embeddings:
        return {
            "clip": clip_label, "gt": gt,
            "correct": False, "wrong": False, "miss": True,
            "frames": total, "faces_detected": 0,
            "votes": {},
        }

    # Vote: match each frame's embedding against gallery
    votes: dict[str, int] = {}
    for emb in all_embeddings:
        name, sim = gallery.match(emb, threshold=match_threshold)
        if name is not None:
            votes[name] = votes.get(name, 0) + 1

    if not votes:
        return {
            "clip": clip_label, "gt": gt,
            "correct": False, "wrong": False, "miss": True,
            "frames": total, "faces_detected": len(all_embeddings),
            "votes": {},
        }

    best_name = max(votes, key=votes.get)
    correct = best_name == gt
    return {
        "clip": clip_label, "gt": gt,
        "correct": correct,
        "wrong": not correct,
        "miss": False,
        "frames": total,
        "faces_detected": len(all_embeddings),
        "votes": votes,
        "best_match": best_name,
        "vote_count": votes[best_name],
    }


def eval_lp_pipeline(
    video_path: Path, clip: dict, device: str,
    model_size: str, output_root: Path,
) -> dict:
    """Run pipeline on LP clip and compare plate to ground truth."""
    clip_label = f"LP-{clip['name']} [{clip['clip_id']}]"
    out_dir = output_root / f"lp_{clip['clip_id']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = analyze_video(
            video_path=str(video_path),
            output_dir=str(out_dir),
            model_family="yolo11",
            model_size=model_size,
            conf_threshold=0.4,
            device=device,
            capture_evidence=False,
            filter_stationary_objects=False,
            min_move_distance=5.0,
            use_reid=False,
        )
    except Exception as e:
        return {"clip": clip_label, "gt": clip["name"], "error": str(e)[:100]}

    gt_plate = clip["name"].strip().upper()
    detected = [
        v.get("plate", "").strip().upper()
        for v in result.get("vehicle_list", [])
        if v.get("plate")
    ]
    best_plate = ""
    best_dist = 1.0
    for p in detected:
        d = 1.0 - SequenceMatcher(None, p, gt_plate).ratio()
        if d < best_dist:
            best_dist = d
            best_plate = p

    return {
        "clip": clip_label,
        "gt": gt_plate,
        "correct": best_plate == gt_plate,
        "best_match": best_plate,
        "edit_distance": round(best_dist, 3),
        "detected_count": len(detected),
        "frames": result.get("total_frames_processed", 0),
    }


# ── Report ──

_global_t0 = time.time()

def print_report(face_results, lp_results, mode: str):
    elapsed = time.time() - _global_t0
    print(f"\n{'=' * 70}")
    print(f"  FANVID EVALUATION — mode={mode}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"{'=' * 70}")

    # ── Face ──
    print(f"\n{'─' * 70}")
    print(f"  FACE RECOGNITION ({len(face_results)} clips)")
    print(f"{'─' * 70}")
    if face_results:
        correct = sum(1 for r in face_results if r.get("correct"))
        wrong = sum(1 for r in face_results if r.get("wrong"))
        missed = sum(1 for r in face_results if r.get("miss"))
        errors = sum(1 for r in face_results if r.get("error"))
        total_valid = len(face_results) - errors

        print(f"  {'Clip':<45s} {'GT':>18s} {'Result':>10s}")
        print(f"  {'─' * 73}")
        for r in face_results:
            if r.get("error"):
                s = "ERROR"
            elif r.get("correct"):
                s = "✓ CORRECT"
            elif r.get("wrong"):
                s = "✗ WRONG"
            else:
                s = "○ MISS"
            label = r["clip"][:43]
            print(f"  {label:<45s} {r['gt'][:18]:>18s} {s:>10s}")

            # Show votes if present
            if r.get("votes"):
                top = sorted(r["votes"].items(), key=lambda x: -x[1])[:3]
                votes_str = ", ".join(f"{n}={c}" for n, c in top)
                print(f"  {'':45s} votes: {votes_str}")
            elif r.get("recognized") and not r.get("correct"):
                print(f"  {'':45s} recognized: {', '.join(r['recognized'][:3])}")

        ar = correct / total_valid * 100 if total_valid else 0
        print(f"\n  Summary ({'pipeline' if mode in ('pipeline','both') else 'direct' }):")
        print(f"    Total:      {len(face_results)}")
        print(f"    Valid:      {total_valid}")
        print(f"    Correct:    {correct} ({ar:.1f}%)")
        print(f"    Wrong:      {wrong}")
        print(f"    Miss:       {missed}")
        print(f"    Errors:     {errors}")
    else:
        print("  No face clips.")

    # ── LP ──
    print(f"\n{'─' * 70}")
    print(f"  LICENSE PLATE ({len(lp_results)} clips)")
    print(f"{'─' * 70}")
    if lp_results:
        correct = sum(1 for r in lp_results if r.get("correct"))
        errors = sum(1 for r in lp_results if r.get("error"))
        print(f"  {'Clip':<45s} {'GT':>15s} {'Detected':>15s} {'Dist':>6s}")
        print(f"  {'─' * 73}")
        for r in lp_results:
            if r.get("error"):
                print(f"  {r['clip'][:43]:<45s} {'ERROR':>36s}")
            else:
                sym = "✓" if r.get("correct") else "✗"
                print(f"  {r['clip'][:43]:<45s} {r['gt'][:15]:>15s} "
                      f"{r['best_match'][:15]:>15s} {r['edit_distance']:>5.2f} {sym}")

        print(f"\n  Summary:")
        print(f"    Total:   {len(lp_results)}")
        print(f"    Correct: {correct} ({correct/len(lp_results)*100:.1f}%)" if lp_results else "")
        print(f"    Errors:  {errors}")
        avg_dist = np.mean([r.get("edit_distance", 1.0) for r in lp_results if not r.get("error")]) if lp_results else 0
        print(f"    Avg ED:  {avg_dist:.3f}")
    else:
        print("  No LP clips.")

    print(f"\n{'=' * 70}\n")

    return {"face_correct": correct if face_results else 0,
            "face_total": len(face_results),
            "lp_correct": correct if 'correct' in dir() else 0,
            "lp_total": len(lp_results)}


# ── main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FANVID evaluation")
    parser.add_argument("--num-videos", type=int, default=5,
                        help="Number of random face clips")
    parser.add_argument("--num-lp", type=int, default=2,
                        help="Number of random LP clips")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-size", default="nano",
                        choices=["nano", "small", "medium", "large", "xlarge"])
    parser.add_argument("--mode", default="direct",
                        choices=["direct", "pipeline", "both"],
                        help="Evaluation mode: 'direct'=insightface on raw frames "
                             "(best for LR); 'pipeline'=full YOLO+track+face pipeline; "
                             "'both'=compare both")
    parser.add_argument("--min-face-size", type=int, default=20,
                        help="Minimum face size in pixels (FANVID LR=180×320, use 20)")
    parser.add_argument("--match-threshold", type=float, default=0.35,
                        help="Cosine similarity threshold for face matching (LR needs lower)")
    parser.add_argument("--face-interval", type=int, default=3,
                        help="Process face every N frames in pipeline mode")
    parser.add_argument("--data-dir", default="data/FANVID")
    parser.add_argument("--output-dir", default="outputs/fanvid_eval")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--max-gallery", type=int, default=None,
                        help="Max mugshots to add to gallery (dev/testing)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  Model: YOLO11{args.model_size}  Mode: {args.mode}")
    print(f"min_face_size={args.min_face_size}  match_threshold={args.match_threshold}")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Download metadata ──
    print("\n[1/5] Downloading FANVID metadata...")
    download_metadata(data_dir)

    mugshot_csv = data_dir / "Celebrity_mugshot.csv"
    celebs_csv = data_dir / "dataset_celebs.csv"
    lp_csv = data_dir / "dataset_lp.csv"
    face_annot_csv = data_dir / "celebrity_annotations_LR.csv"
    lp_annot_csv = data_dir / "license_plate_annotations_LR.csv"

    # ── 2. Build face gallery ──
    print("\n[2/5] Building face embedding gallery from HR mugshots...")
    fr = FaceRecognizer(
        gallery_path="face_gallery.json",
        device=device,
        capture_unknowns=False,
    )
    if fr.available:
        build_face_gallery(fr, mugshot_csv)
    else:
        print("  insightface unavailable — face eval disabled")

    # ── 3. Select clips ──
    print(f"\n[3/5] Selecting clips...")
    rng = random.Random(args.seed)

    face_clips_all = load_dataset(celebs_csv)
    # Prioritize test split
    test_face = [c for c in face_clips_all if c["split"] == "test"]
    if not test_face:
        test_face = face_clips_all
    selected_face = rng.sample(test_face, min(args.num_videos, len(test_face)))

    lp_clips_all = load_dataset(lp_csv)
    test_lp = [c for c in lp_clips_all if c["split"] == "test"]
    if not test_lp:
        test_lp = lp_clips_all[:args.num_lp]
    selected_lp = rng.sample(test_lp, min(args.num_lp, len(test_lp)))

    print(f"  {len(selected_face)} face clips, {len(selected_lp)} LP clips")

    # ── 4. Evaluate ──
    print("\n[4/5] Downloading and evaluating...")
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    face_results = []
    for clip in tqdm(selected_face, desc="Face clips"):
        if not args.skip_download:
            vpath = download_clip(clip, videos_dir)
        else:
            vpath = videos_dir / f"clip_{clip['clip_id']}_{clip['video_id']}.mp4"
        if vpath is None or not vpath.exists():
            log.warning("  Skip %s — no video", clip["name"])
            continue
        vpath = _ensure_good_video(vpath)
        if vpath is None:
            log.warning("  Skip %s — undecodable", clip["name"])
            continue

        if args.mode in ("pipeline", "both"):
            r = eval_face_pipeline(
                vpath, clip, device, args.model_size, out_dir,
                args.min_face_size, args.face_interval,
            )
            face_results.append(r)
            _show_face_status(r)
        if args.mode in ("direct", "both"):
            r = eval_face_direct(
                vpath, clip, fr.gallery,
                args.min_face_size, args.match_threshold,
            )
            face_results.append(r)
            _show_face_status(r)
        if args.mode == "both":
            face_results.append({})  # separator

    lp_results = []
    for clip in tqdm(selected_lp, desc="LP clips"):
        if not args.skip_download:
            vpath = download_clip(clip, videos_dir)
        else:
            vpath = videos_dir / f"clip_{clip['clip_id']}_{clip['video_id']}.mp4"
        if vpath is None or not vpath.exists():
            continue
        vpath = _ensure_good_video(vpath)
        if vpath is None:
            continue

        r = eval_lp_pipeline(vpath, clip, device, args.model_size, out_dir)
        lp_results.append(r)
        sym = "✓" if r.get("correct") else ("!" if r.get("error") else "✗")
        det = r.get("best_match", "—")[:12]
        print(f"  {sym} {r['clip'][:45]:<45s} GT={r['gt'][:12]:>12s} DET={det}")

    # ── 5. Report ──
    print("\n[5/5] Report")
    report = {
        "mode": args.mode,
        "model_size": args.model_size,
        "device": device,
        "min_face_size": args.min_face_size,
        "match_threshold": args.match_threshold,
        "face_results": face_results,
        "lp_results": lp_results,
    }
    report_path = out_dir / "eval_results.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Results → {report_path}")


def _show_face_status(r: dict):
    if r.get("error"):
        sym = "!"
    elif r.get("correct"):
        sym = "✓"
    elif r.get("wrong"):
        sym = "✗"
    else:
        sym = "○"
    label = r.get("clip", "?")[:50]
    print(f"  {sym} {label:<52s} GT={r.get('gt','')[:20]}")


if __name__ == "__main__":
    main()
