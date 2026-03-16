#!/usr/bin/env python3
"""
Path-Based Formula Detector
============================
Detects formulaic openers by comparing their word-by-word embedding PATH
traversal to other openers in the batch. No template library, no hard-coded
strings — pure path geometry.

The idea: formulaic openers all take the SAME journey through embedding space.
"X is just Y with Z" openers all have: topic → connector → reframe → qualifier.
Fresh openers each have a unique path signature.

We represent each opener as a "path fingerprint" — the sequence of direction
vectors between consecutive word embeddings — then check if that fingerprint
is too similar to the rest of the batch.

Usage:
    python3 -u path_formula_detector.py          # Analyze + validate
    python3 -u path_formula_detector.py --batch   # Score a batch from stdin (JSON array)
"""

import json
import sys
import glob
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

SRC_DIR = Path(__file__).parent
EVAL_GLOB = str(SRC_DIR / "eval_results_gen_*.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_all_eval_data():
    files = sorted(glob.glob(EVAL_GLOB))
    seen = set()
    openers = []
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
            openers.append({
                "text": r["comment"],
                "freshness": r["scores"]["formula_freshness"]["score"],
                "warmth": r["scores"]["warmth_signal"]["score"],
                "avg": np.mean([r["scores"][d]["score"] for d in
                    ["reply_likelihood", "humor_quality", "creativity", "specificity",
                     "human_likeness", "closer_strength", "warmth_signal", "formula_freshness"]]),
            })
    return openers


def compute_path_fingerprint(text, model, embed_dim=384):
    """
    Compute the path fingerprint of an opener.

    Returns a dict with multiple path representations:
    - step_vectors: list of direction vectors between consecutive word embeddings
    - avg_direction: average of all step vectors (the "average journey direction")
    - step_magnitudes: list of step sizes (cosine distances)
    - curvature: list of cosine sims between consecutive steps (how much the path bends)
    - path_embedding: a fixed-dim representation of the entire path (for comparison)
    """
    words = text.lower().split()
    if len(words) < 3:
        return None

    word_embs = model.encode(words, normalize_embeddings=True)

    # Step vectors: direction of travel between consecutive words
    step_vectors = []
    for i in range(len(word_embs) - 1):
        step = word_embs[i + 1] - word_embs[i]
        norm = np.linalg.norm(step)
        if norm > 0:
            step = step / norm  # normalize direction
        step_vectors.append(step)

    step_vectors = np.array(step_vectors)

    # Average direction: the "average journey" through embedding space
    avg_direction = step_vectors.mean(axis=0)
    avg_norm = np.linalg.norm(avg_direction)
    if avg_norm > 0:
        avg_direction = avg_direction / avg_norm

    # Step magnitudes (cosine distances between consecutive words)
    step_magnitudes = []
    for i in range(len(word_embs) - 1):
        d = cosine_dist(word_embs[i], word_embs[i + 1])
        step_magnitudes.append(d)

    # Curvature: how much the path bends at each point
    curvature = []
    for i in range(len(step_vectors) - 1):
        sim = 1.0 - cosine_dist(step_vectors[i], step_vectors[i + 1])
        curvature.append(sim)

    # Path embedding: concatenate [avg_direction, step_stats, curvature_stats]
    # This gives a fixed-dim representation regardless of opener length
    step_mag_arr = np.array(step_magnitudes)
    curv_arr = np.array(curvature) if curvature else np.array([0.0])

    path_stats = np.array([
        step_mag_arr.mean(),
        step_mag_arr.std(),
        step_mag_arr.max(),
        step_mag_arr[-1],  # final step (closer surprise)
        curv_arr.mean(),
        curv_arr.std(),
        curv_arr.min(),  # sharpest turn
        len(words),
    ])

    # The path embedding: avg_direction (384d) + stats (8d) = 392d
    path_embedding = np.concatenate([avg_direction, path_stats])

    return {
        "step_vectors": step_vectors,
        "avg_direction": avg_direction,
        "step_magnitudes": step_magnitudes,
        "curvature": curvature,
        "path_embedding": path_embedding,
        "path_stats": path_stats,
    }


def compute_path_similarity(fp1, fp2):
    """
    Compute similarity between two path fingerprints.
    Returns multiple similarity measures.
    """
    # 1. Average direction similarity (do they go the same way?)
    dir_sim = 1.0 - cosine_dist(fp1["avg_direction"], fp2["avg_direction"])

    # 2. Full path embedding similarity
    emb_sim = 1.0 - cosine_dist(fp1["path_embedding"], fp2["path_embedding"])

    # 3. Step vector sequence similarity (DTW-lite: compare overlapping steps)
    steps1 = fp1["step_vectors"]
    steps2 = fp2["step_vectors"]
    min_len = min(len(steps1), len(steps2))
    if min_len > 0:
        step_sims = [1.0 - cosine_dist(steps1[i], steps2[i]) for i in range(min_len)]
        seq_sim = np.mean(step_sims)
    else:
        seq_sim = 0.0

    return {
        "direction_sim": dir_sim,
        "embedding_sim": emb_sim,
        "sequence_sim": seq_sim,
        "combined": 0.4 * dir_sim + 0.4 * emb_sim + 0.2 * seq_sim,
    }


def score_batch(texts, model):
    """
    Score a batch of openers by their path similarity to each other.
    Returns a "conformity score" for each opener — how similar its path
    is to the average path of all other openers in the batch.

    High conformity = formulaic (same path as everyone else)
    Low conformity = fresh (unique path)
    """
    fingerprints = []
    valid_indices = []

    for i, text in enumerate(texts):
        fp = compute_path_fingerprint(text, model)
        if fp is not None:
            fingerprints.append(fp)
            valid_indices.append(i)

    n = len(fingerprints)
    if n < 3:
        return {i: {"conformity": 0.0, "is_formulaic": False} for i in range(len(texts))}

    # Compute pairwise path similarity matrix
    dir_sims = np.zeros((n, n))
    emb_sims = np.zeros((n, n))

    # Extract all avg_directions and path_embeddings for vectorized computation
    all_dirs = np.array([fp["avg_direction"] for fp in fingerprints])
    all_embs = np.array([fp["path_embedding"][:384] for fp in fingerprints])  # just the direction part
    all_embs_norm = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-10)
    all_dirs_norm = all_dirs / (np.linalg.norm(all_dirs, axis=1, keepdims=True) + 1e-10)

    # Pairwise cosine similarities
    dir_sims = all_dirs_norm @ all_dirs_norm.T
    emb_sims = all_embs_norm @ all_embs_norm.T

    # For each opener, compute its average similarity to all others
    results = {}
    for idx_in_batch in range(len(texts)):
        if idx_in_batch not in valid_indices:
            results[idx_in_batch] = {"conformity": 0.0, "is_formulaic": False}
            continue

        i = valid_indices.index(idx_in_batch)

        # Average direction similarity to all other openers (excluding self)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        avg_dir_sim = dir_sims[i, mask].mean()
        max_dir_sim = dir_sims[i, mask].max()
        avg_emb_sim = emb_sims[i, mask].mean()

        # Conformity score: how much does this path look like the average path?
        conformity = 0.5 * avg_dir_sim + 0.3 * max_dir_sim + 0.2 * avg_emb_sim

        # Find most similar opener
        other_sims = dir_sims[i, mask]
        most_similar_idx = valid_indices[np.where(mask)[0][other_sims.argmax()]]

        results[idx_in_batch] = {
            "conformity": float(conformity),
            "avg_dir_sim": float(avg_dir_sim),
            "max_dir_sim": float(max_dir_sim),
            "avg_emb_sim": float(avg_emb_sim),
            "most_similar_idx": most_similar_idx,
            "most_similar_sim": float(max_dir_sim),
        }

    return results


def main():
    print("=" * 80)
    print("PATH-BASED FORMULA DETECTOR")
    print("=" * 80)
    print("Detecting formulaic openers by comparing path traversal patterns.")
    print("No template library. No hard-coded strings. Pure path geometry.\n")

    openers = load_all_eval_data()
    print(f"Loaded {len(openers)} unique scored openers")

    model = SentenceTransformer(EMBEDDING_MODEL)

    # Compute all fingerprints
    print("Computing path fingerprints...")
    fingerprints = []
    valid = []
    for i, o in enumerate(openers):
        fp = compute_path_fingerprint(o["text"], model)
        if fp is not None:
            fingerprints.append(fp)
            valid.append(i)

    print(f"  {len(fingerprints)} valid fingerprints")

    n = len(fingerprints)
    freshness = np.array([openers[valid[i]]["freshness"] for i in range(n)])
    avg_scores = np.array([openers[valid[i]]["avg"] for i in range(n)])

    # ════════════════════════════════════════════════════════════
    # 1. PATH DIRECTION SIMILARITY MATRIX
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("1. PATH DIRECTION ANALYSIS")
    print("=" * 80)

    all_dirs = np.array([fp["avg_direction"] for fp in fingerprints])
    all_dirs_norm = all_dirs / (np.linalg.norm(all_dirs, axis=1, keepdims=True) + 1e-10)
    dir_sim_matrix = all_dirs_norm @ all_dirs_norm.T

    # For each opener: average path direction similarity to all others
    avg_dir_sims = np.zeros(n)
    max_dir_sims = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        avg_dir_sims[i] = dir_sim_matrix[i, mask].mean()
        max_dir_sims[i] = dir_sim_matrix[i, mask].max()

    # Correlation with freshness
    r_avg, p_avg = spearmanr(avg_dir_sims, freshness)
    r_max, p_max = spearmanr(max_dir_sims, freshness)
    print(f"\n  avg_path_direction_sim vs freshness: r={r_avg:+.4f} p={p_avg:.6f}")
    print(f"  max_path_direction_sim vs freshness: r={r_max:+.4f} p={p_max:.6f}")

    # Compare formulaic vs fresh groups
    form_mask = freshness <= 4
    fresh_mask = freshness >= 8

    if form_mask.sum() >= 3 and fresh_mask.sum() >= 3:
        form_avg = avg_dir_sims[form_mask].mean()
        fresh_avg = avg_dir_sims[fresh_mask].mean()
        print(f"\n  Formulaic openers avg path conformity: {form_avg:.4f}")
        print(f"  Fresh openers avg path conformity:     {fresh_avg:.4f}")
        print(f"  Gap: {form_avg - fresh_avg:+.4f} (positive = formulaic more similar to each other)")

    # ════════════════════════════════════════════════════════════
    # 2. CURVATURE ANALYSIS
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("2. PATH CURVATURE ANALYSIS")
    print("=" * 80)

    avg_curvatures = np.array([np.mean(fp["curvature"]) if fp["curvature"] else 0 for fp in fingerprints])
    std_curvatures = np.array([np.std(fp["curvature"]) if len(fp["curvature"]) > 1 else 0 for fp in fingerprints])
    min_curvatures = np.array([np.min(fp["curvature"]) if fp["curvature"] else 0 for fp in fingerprints])

    for name, vals in [("avg_curvature", avg_curvatures), ("std_curvature", std_curvatures), ("min_curvature", min_curvatures)]:
        r, p = spearmanr(vals, freshness)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        form_v = vals[form_mask].mean() if form_mask.sum() > 0 else 0
        fresh_v = vals[fresh_mask].mean() if fresh_mask.sum() > 0 else 0
        print(f"  {name:20s}: r={r:+.4f} p={p:.6f} {sig:3s}  form={form_v:.4f} fresh={fresh_v:.4f}")

    # ════════════════════════════════════════════════════════════
    # 3. STEP VECTOR DIVERSITY (within each opener)
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("3. INTERNAL STEP DIVERSITY")
    print("=" * 80)

    internal_diversities = []
    for fp in fingerprints:
        steps = fp["step_vectors"]
        if len(steps) < 2:
            internal_diversities.append(0)
            continue
        # Pairwise similarity between step vectors within this opener
        steps_norm = steps / (np.linalg.norm(steps, axis=1, keepdims=True) + 1e-10)
        step_sim = steps_norm @ steps_norm.T
        # Average off-diagonal similarity
        mask = ~np.eye(len(steps), dtype=bool)
        internal_diversities.append(1.0 - step_sim[mask].mean())  # diversity = 1 - similarity

    internal_diversities = np.array(internal_diversities)
    r, p = spearmanr(internal_diversities, freshness)
    print(f"\n  internal_step_diversity vs freshness: r={r:+.4f} p={p:.6f}")
    print(f"  Formulaic avg internal diversity: {internal_diversities[form_mask].mean():.4f}")
    print(f"  Fresh avg internal diversity:     {internal_diversities[fresh_mask].mean():.4f}")

    # ════════════════════════════════════════════════════════════
    # 4. CONFORMITY SCORE (the final metric)
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("4. CONFORMITY SCORE OPTIMIZATION")
    print("=" * 80)

    # Features:
    # f1 = avg_dir_sim (how similar is your path direction to everyone else's)
    # f2 = max_dir_sim (how similar is your path to your closest neighbor)
    # f3 = 1 - internal_diversity (low internal diversity = monotone path)
    # f4 = avg_curvature (how much your path bends — formulaic = predictable bending)

    features = {
        "avg_dir_sim": avg_dir_sims,
        "max_dir_sim": max_dir_sims,
        "low_internal_div": 1.0 - internal_diversities,
        "avg_curvature": avg_curvatures,
    }

    print("\n  Individual feature correlations with freshness:")
    for name, vals in features.items():
        r, p = spearmanr(vals, freshness)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {name:20s}: r={r:+.4f} p={p:.6f} {sig}")

    # Normalize features to [0, 1]
    norm_features = {}
    norm_params = {}
    for name, vals in features.items():
        vmin, vmax = vals.min(), vals.max()
        norm_params[name] = {"min": float(vmin), "max": float(vmax)}
        norm_features[name] = np.clip((vals - vmin) / (vmax - vmin + 1e-10), 0, 1)

    feature_matrix = np.column_stack([norm_features[k] for k in features])
    feature_names = list(features.keys())
    is_formulaic = freshness <= 4

    # Grid search for optimal weights
    print("\n  Grid searching weights...")
    best_f1 = 0
    best_weights = None
    best_threshold = 0

    weight_options = np.linspace(0, 1, 11)
    for w1 in weight_options:
        for w2 in weight_options:
            for w3 in weight_options:
                for w4 in weight_options:
                    total = w1 + w2 + w3 + w4
                    if total < 0.01:
                        continue
                    w = np.array([w1, w2, w3, w4]) / total
                    scores = feature_matrix @ w

                    for pct in range(40, 95, 5):
                        thresh = np.percentile(scores, pct)
                        pred = scores >= thresh
                        tp = (pred & is_formulaic).sum()
                        fp = (pred & ~is_formulaic).sum()
                        fn = (~pred & is_formulaic).sum()
                        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                        if f1 > best_f1:
                            best_f1 = f1
                            best_weights = w.copy()
                            best_threshold = thresh

    print(f"\n  *** OPTIMAL WEIGHTS ***")
    for name, w in zip(feature_names, best_weights):
        print(f"    w_{name} = {w:.4f}")

    final_scores = feature_matrix @ best_weights
    pred = final_scores >= best_threshold
    tp = (pred & is_formulaic).sum()
    fp = (pred & ~is_formulaic).sum()
    fn = (~pred & is_formulaic).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"\n  Threshold: {best_threshold:.4f}")
    print(f"  F1={best_f1:.3f} P={prec:.3f} R={rec:.3f} TP={tp} FP={fp} FN={fn}")

    comp_r, comp_p = spearmanr(final_scores, freshness)
    print(f"  Correlation with freshness: r={comp_r:+.4f} p={comp_p:.8f}")

    # Operating points
    print(f"\n  Operating points:")
    print(f"  {'Threshold':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    for pct in [40, 50, 60, 70, 80, 85, 90]:
        t = np.percentile(final_scores, pct)
        p2 = final_scores >= t
        tp2 = (p2 & is_formulaic).sum()
        fp2 = (p2 & ~is_formulaic).sum()
        fn2 = (~p2 & is_formulaic).sum()
        pr = tp2/(tp2+fp2) if (tp2+fp2) > 0 else 0
        rc = tp2/(tp2+fn2) if (tp2+fn2) > 0 else 0
        f1_ = 2*pr*rc/(pr+rc) if (pr+rc) > 0 else 0
        print(f"  {t:10.4f} {pr:10.3f} {rc:10.3f} {f1_:10.3f}")

    # ════════════════════════════════════════════════════════════
    # 5. EXAMPLES
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("5. EXAMPLES")
    print("=" * 80)

    sorted_idx = np.argsort(final_scores)[::-1]

    print("\n  MOST CONFORMIST paths (highest conformity = most formulaic):")
    for rank, i in enumerate(sorted_idx[:12]):
        oi = valid[i]
        o = openers[oi]
        print(f"    {rank+1:2d}. score={final_scores[i]:.3f} fresh={o['freshness']} avg_dir={avg_dir_sims[i]:.3f} max_dir={max_dir_sims[i]:.3f}")
        print(f"        \"{o['text'][:85]}\"")

    print("\n  MOST UNIQUE paths (lowest conformity = most fresh):")
    for rank, i in enumerate(sorted_idx[-10:]):
        oi = valid[i]
        o = openers[oi]
        print(f"    {rank+1:2d}. score={final_scores[i]:.3f} fresh={o['freshness']} avg_dir={avg_dir_sims[i]:.3f}")
        print(f"        \"{o['text'][:85]}\"")

    # Score distribution by freshness
    print(f"\n  Score distribution by freshness:")
    for fv in sorted(set(freshness)):
        mask = freshness == fv
        if mask.sum() < 2:
            continue
        vals = final_scores[mask]
        print(f"    freshness={int(fv):2d}: n={mask.sum():4d}  mean={vals.mean():.3f}  std={vals.std():.3f}")

    # ════════════════════════════════════════════════════════════
    # 6. FINAL FORMULA
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("FINAL FORMULA — PURE PATH GEOMETRY")
    print("=" * 80)
    print(f"""
  For a batch of openers [T1, T2, ..., Tn]:

  1. For each opener Ti, embed each word -> sequence of word vectors
  2. Compute step vectors: step_j = word[j+1] - word[j] (normalized)
  3. Compute avg_direction = mean(step_vectors) (normalized)
  4. Compute internal_diversity = 1 - mean(pairwise_sim(step_vectors))

  5. For each Ti, compute:
     f1 = avg cosine_sim(avg_direction_i, avg_direction_j) for all j != i
          (how similar is your path to everyone else's path)
     f2 = max cosine_sim(avg_direction_i, avg_direction_j) for all j != i
          (how close is your nearest path neighbor)
     f3 = 1 - internal_diversity
          (how monotone is your path — low diversity = repetitive structure)
     f4 = avg curvature (mean cosine_sim between consecutive step vectors)

  6. conformity_score = {best_weights[0]:.2f}*f1 + {best_weights[1]:.2f}*f2 + {best_weights[2]:.2f}*f3 + {best_weights[3]:.2f}*f4

  7. REJECT if conformity_score >= {best_threshold:.4f}

  This requires NO template library, NO hard-coded strings.
  A formulaic opener is one whose path looks like everyone else's path.
""")

    # Save model
    model_data = {
        "version": 3,
        "type": "path_conformity",
        "description": "Detects formulaic openers by path traversal similarity. No templates, no string matching.",
        "embedding_model": EMBEDDING_MODEL,
        "feature_names": feature_names,
        "weights": best_weights.tolist(),
        "normalization": norm_params,
        "threshold": float(best_threshold),
        "metrics": {"f1": float(best_f1), "precision": float(prec), "recall": float(rec)},
        "correlation_with_freshness": float(comp_r),
    }

    model_path = SRC_DIR / "path_formula_model.json"
    with open(model_path, "w") as f:
        json.dump(model_data, f, indent=2)
    print(f"  Model saved to: {model_path}")


if __name__ == "__main__":
    main()
