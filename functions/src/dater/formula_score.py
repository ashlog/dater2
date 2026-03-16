#!/usr/bin/env python3
"""
Formula Direction Scorer
========================
Scores openers by projection onto the learned formulaic↔fresh direction vector.
No templates, no string matching — pure embedding geometry.

Usage:
    # Score a batch (JSON array on stdin, JSON array on stdout)
    echo '["opener 1", "opener 2"]' | python3 formula_score.py

    # Train/update the direction vector from eval data
    python3 formula_score.py --train
"""

import json
import sys
import glob
import numpy as np
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent  # src/dater/.. = src/
VECTOR_FILE = SRC_DIR / "formula_direction.json"
EVAL_GLOB = str(SRC_DIR / "eval_results_gen_*.json")


def train():
    """Learn the formula direction vector from scored eval data."""
    from sentence_transformers import SentenceTransformer
    import warnings; warnings.filterwarnings('ignore')

    files = sorted(glob.glob(EVAL_GLOB))
    seen = set()
    texts, freshness_scores = [], []

    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        for r in data["results"]:
            if "formula_freshness" not in r["scores"]:
                continue
            key = r["comment"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            texts.append(r["comment"])
            freshness_scores.append(r["scores"]["formula_freshness"]["score"])

    freshness = np.array(freshness_scores)
    print(f"Training on {len(texts)} openers", file=sys.stderr)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    form_mask = freshness <= 4
    fresh_mask = freshness >= 8

    c_form = embeddings[form_mask].mean(axis=0)
    c_form /= np.linalg.norm(c_form)
    c_fresh = embeddings[fresh_mask].mean(axis=0)
    c_fresh /= np.linalg.norm(c_fresh)

    direction = c_form - c_fresh
    direction /= np.linalg.norm(direction)

    # Compute normalization bounds
    projections = embeddings @ direction

    model_data = {
        "direction": direction.tolist(),
        "proj_min": float(projections.min()),
        "proj_max": float(projections.max()),
        "n_train": len(texts),
        "n_formulaic": int(form_mask.sum()),
        "n_fresh": int(fresh_mask.sum()),
    }

    with open(VECTOR_FILE, "w") as f:
        json.dump(model_data, f)

    print(f"Saved direction vector to {VECTOR_FILE}", file=sys.stderr)
    return model_data


def score_batch(texts):
    """Score a batch of openers. Returns list of {score, is_formulaic} dicts."""
    from sentence_transformers import SentenceTransformer
    import warnings; warnings.filterwarnings('ignore')

    if not VECTOR_FILE.exists():
        train()

    with open(VECTOR_FILE) as f:
        md = json.load(f)

    direction = np.array(md["direction"])
    p_min, p_max = md["proj_min"], md["proj_max"]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    projections = embeddings @ direction
    normalized = np.clip((projections - p_min) / (p_max - p_min), 0, 1)

    results = []
    for i, text in enumerate(texts):
        results.append({
            "formula_proj": float(normalized[i]),
            "is_formulaic": float(normalized[i]) > 0.70,
        })

    return results


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        train()
    else:
        texts = json.loads(sys.stdin.read())
        results = score_batch(texts)
        print(json.dumps(results))
