#!/usr/bin/env python3
"""Batch-match unknown faces against a known-face database.

Usage:
  # Match unknown faces against a directory of photos
  python scripts/face_db_matcher.py --db photos/ --threshold 0.40

  # Match and auto-promote high-confidence matches to gallery
  python scripts/face_db_matcher.py --db photos/ --promote --min-conf 0.55

  # List all unknown faces with their current match status
  python scripts/face_db_matcher.py --list
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analytics.face_recognition import FaceRecognizer


def load_db_faces(db_dir: str) -> list[tuple[str, np.ndarray]]:
    """Load all images from a directory, return [(name, image_rgb)]."""
    known = []
    for path in sorted(Path(db_dir).glob("*")):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        img = cv2.imread(str(path))
        if img is None:
            print(f"  WARNING: cannot read {path}")
            continue
        name = path.stem  # filename without extension = person name
        known.append((name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return known


def main():
    parser = argparse.ArgumentParser(description="Face database matcher")
    parser.add_argument("--db", default=None,
                        help="Directory of known-face photos (filename = person name)")
    parser.add_argument("--threshold", type=float, default=0.40,
                        help="Cosine similarity threshold (default: 0.40)")
    parser.add_argument("--promote", action="store_true",
                        help="Auto-promote matches above --min-conf to gallery")
    parser.add_argument("--min-conf", type=float, default=0.55,
                        help="Minimum confidence for auto-promote")
    parser.add_argument("--list", action="store_true",
                        help="Just list unknown faces with match status")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    fr = FaceRecognizer(device=args.device, capture_unknowns=False)

    if not fr.available:
        print("ERROR: insightface not available. Install with: pip install insightface")
        sys.exit(1)

    unknowns = fr.unknown_store.list_unmatched()
    if not unknowns:
        print("No unmatched unknown faces found.")
        # Show all known
        known_names = fr.gallery.known_names()
        if known_names:
            print(f"Gallery has {len(known_names)} known faces:")
            for n in known_names:
                print(f"  - {n}")
        return

    print(f"Unknown faces in store: {fr.unknown_store.count()}")
    print(f"  Unmatched: {len(unknowns)}")
    print(f"  Matched:   {fr.unknown_store.count() - len(unknowns)}")

    # ---- Just list ----
    if args.list and not args.db:
        for u in unknowns:
            m = f"  {u['id']:>15s}  track {u['track_id']:>4d}  frame {u['frame']:>5d}  score={u['det_score']:.2f}  {u['image']}"
            print(m)
        return

    # ---- Batch match ----
    if not args.db:
        print("Use --db to specify a directory of known-face photos, or --list to list unknowns.")
        return

    db_faces = load_db_faces(args.db)
    if not db_faces:
        print(f"No face images found in {args.db}")
        sys.exit(1)

    print(f"\nLoaded {len(db_faces)} known faces from {args.db}:")
    for name, _ in db_faces:
        print(f"  - {name}")

    print(f"\n{'=' * 70}")
    print(f"  BATCH MATCHING {len(unknowns)} UNKNOWNS × {len(db_faces)} DB FACES")
    print(f"{'=' * 70}")

    # Pre-compute DB embeddings for speed
    print("\nComputing DB embeddings...")
    db_embeddings: list[tuple[str, np.ndarray]] = []
    for name, img in db_faces:
        faces = fr._model.get(img)
        if not faces:
            print(f"  WARNING: no face detected in {name}")
            continue
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        if face.normed_embedding is not None:
            db_embeddings.append((name, face.normed_embedding))
    print(f"  Computed {len(db_embeddings)} embeddings")

    # Match each unknown against all DB faces
    matches_found = 0
    for u in unknowns:
        emb = u["embedding"]
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)

        best_name = None
        best_sim = 0.0
        for db_name, db_emb in db_embeddings:
            sim = float(np.dot(emb_norm, db_emb))
            sim = max(-1.0, min(1.0, sim))
            if sim > best_sim:
                best_sim = sim
                best_name = db_name

        if best_sim >= args.threshold:
            matches_found += 1
            status = "  MATCH"
            extra = ""
            if args.promote and best_sim >= args.min_conf:
                fr.gallery.add(best_name, emb)
                fr.unknown_store.mark_matched(u["id"], best_name, best_sim)
                status = "  PROMOTED"
                extra = " → added to gallery"
            print(f"{status}  {u['id']:>15s} → {best_name:>20s}  (sim={best_sim:.3f}){extra}")
        else:
            print(f"  NO MATCH  {u['id']:>15s}  best={best_name or '?'}  (sim={best_sim:.3f})")

    print(f"\n{matches_found}/{len(unknowns)} unknowns matched above threshold ({args.threshold})")

    # Summary
    known_after = len(fr.gallery.known_names())
    unknown_remaining = len(fr.unknown_store.list_unmatched())
    print(f"\nGallery: {known_after} known faces")
    print(f"Unknown store: {unknown_remaining} unmatched remaining")


if __name__ == "__main__":
    main()
