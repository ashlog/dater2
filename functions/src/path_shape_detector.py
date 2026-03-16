#!/usr/bin/env python3
"""
Path Shape Formula Detector
=============================
Detects formulaic openers by comparing the SHAPE of their embedding path
traversal — not the direction, not template matching.

Key insight: "X is just Y with Z" openers about food vs music go in different
DIRECTIONS but have the same SHAPE: medium step → small connector → BIG reframe → question.

We capture shape via:
1. Step size rhythm (normalized step magnitudes)
2. Position of max jump (where the surprise happens)
3. Curvature rhythm (how the path bends)
4. Then compare these shape signatures across all openers in the batch

Two openers with the same shape = same formula, regardless of topic.
"""

import json
import sys
import glob
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import spearmanr
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

SRC_DIR = Path(__file__).parent
EVAL_GLOB = str(SRC_DIR / "eval_results_gen_*.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SHAPE_DIM = 12  # fixed-length shape vector (interpolated from variable-length paths)


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
                "avg": np.mean([r["scores"][d]["score"] for d in
                    ["reply_likelihood", "humor_quality", "creativity", "specificity",
                     "human_likeness", "closer_strength", "warmth_signal", "formula_freshness"]]),
            })
    return openers


def compute_shape(text, model):
    """
    Compute the shape signature of an opener's path through embedding space.
    Returns a fixed-length vector capturing HOW the path moves, not WHERE.
    """
    words = text.lower().split()
    if len(words) < 4:
        return None

    word_embs = model.encode(words, normalize_embeddings=True)
    n_steps = len(word_embs) - 1

    # Step sizes (cosine distances between consecutive words)
    step_sizes = np.array([cosine_dist(word_embs[i], word_embs[i+1]) for i in range(n_steps)])

    # Curvature at each interior point (cosine sim between consecutive step directions)
    step_dirs = np.array([word_embs[i+1] - word_embs[i] for i in range(n_steps)])
    step_dir_norms = np.linalg.norm(step_dirs, axis=1, keepdims=True)
    step_dirs_normalized = step_dirs / (step_dir_norms + 1e-10)

    curvatures = np.array([
        1.0 - cosine_dist(step_dirs_normalized[i], step_dirs_normalized[i+1])
        for i in range(len(step_dirs_normalized) - 1)
    ])

    # Normalize step sizes to sum to 1 (the "rhythm")
    step_rhythm = step_sizes / (step_sizes.sum() + 1e-10)

    # Interpolate to fixed length SHAPE_DIM for comparison
    positions = np.linspace(0, 1, len(step_rhythm))
    target_positions = np.linspace(0, 1, SHAPE_DIM)

    if len(step_rhythm) >= 2:
        f_rhythm = interp1d(positions, step_rhythm, kind='linear', fill_value='extrapolate')
        shape_rhythm = f_rhythm(target_positions)
    else:
        shape_rhythm = np.ones(SHAPE_DIM) / SHAPE_DIM

    if len(curvatures) >= 2:
        curv_positions = np.linspace(0, 1, len(curvatures))
        f_curv = interp1d(curv_positions, curvatures, kind='linear', fill_value='extrapolate')
        shape_curvature = f_curv(target_positions)
    else:
        shape_curvature = np.zeros(SHAPE_DIM)

    # Shape features (topic-independent)
    max_jump_pos = float(np.argmax(step_sizes)) / n_steps  # where in [0,1] is the biggest jump
    final_ratio = step_sizes[-1] / (step_sizes.max() + 1e-10)  # is the biggest jump at the end?
    step_entropy = float(-np.sum(step_rhythm * np.log(step_rhythm + 1e-10)))  # predictability of rhythm
    step_autocorr = float(np.corrcoef(step_sizes[:-1], step_sizes[1:])[0, 1]) if len(step_sizes) > 2 else 0

    # The full shape vector: rhythm (12d) + curvature (12d) + scalar features (4d) = 28d
    shape_vector = np.concatenate([
        shape_rhythm,
        shape_curvature,
        [max_jump_pos, final_ratio, step_entropy, step_autocorr],
    ])

    return {
        "shape_vector": shape_vector,
        "shape_rhythm": shape_rhythm,
        "shape_curvature": shape_curvature,
        "step_sizes": step_sizes,
        "max_jump_pos": max_jump_pos,
        "final_ratio": final_ratio,
        "step_entropy": step_entropy,
        "step_autocorr": step_autocorr,
        "word_count": len(words),
    }


def main():
    print("=" * 80)
    print("PATH SHAPE FORMULA DETECTOR")
    print("=" * 80)
    print("Compares the SHAPE of each opener's embedding path to all others.")
    print("Same shape = same formula, regardless of topic.\n")

    openers = load_all_eval_data()
    print(f"Loaded {len(openers)} unique scored openers")

    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Computing path shapes...")
    shapes = []
    valid = []
    for i, o in enumerate(openers):
        s = compute_shape(o["text"], model)
        if s is not None:
            shapes.append(s)
            valid.append(i)

    n = len(shapes)
    print(f"  {n} valid shapes (shape_vector dim = {shapes[0]['shape_vector'].shape[0]})")

    freshness = np.array([openers[valid[i]]["freshness"] for i in range(n)])
    is_formulaic = freshness <= 4
    is_fresh = freshness >= 8

    # ════════════════════════════════════════════════════════════
    # 1. SCALAR SHAPE FEATURES
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("1. SCALAR SHAPE FEATURES vs FRESHNESS")
    print("=" * 80)

    scalar_features = {
        "max_jump_position": np.array([s["max_jump_pos"] for s in shapes]),
        "final_ratio": np.array([s["final_ratio"] for s in shapes]),
        "step_entropy": np.array([s["step_entropy"] for s in shapes]),
        "step_autocorrelation": np.array([s["step_autocorr"] for s in shapes]),
    }

    for name, vals in scalar_features.items():
        r, p = spearmanr(vals, freshness)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        form_mean = vals[is_formulaic].mean() if is_formulaic.sum() > 0 else 0
        fresh_mean = vals[is_fresh].mean() if is_fresh.sum() > 0 else 0
        print(f"  {name:25s}: r={r:+.4f} p={p:.6f} {sig:3s}  formulaic={form_mean:.4f} fresh={fresh_mean:.4f} gap={form_mean-fresh_mean:+.4f}")

    # ════════════════════════════════════════════════════════════
    # 2. SHAPE VECTOR SIMILARITY (the main event)
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("2. SHAPE VECTOR CONFORMITY")
    print("=" * 80)

    shape_matrix = np.array([s["shape_vector"] for s in shapes])
    # L2-normalize for cosine similarity
    shape_norms = np.linalg.norm(shape_matrix, axis=1, keepdims=True)
    shape_normed = shape_matrix / (shape_norms + 1e-10)

    # Pairwise shape similarity
    shape_sim_matrix = shape_normed @ shape_normed.T

    # Per-opener conformity metrics
    avg_shape_sim = np.zeros(n)
    max_shape_sim = np.zeros(n)
    top5_shape_sim = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        sims = shape_sim_matrix[i, mask]
        avg_shape_sim[i] = sims.mean()
        max_shape_sim[i] = sims.max()
        top5_shape_sim[i] = np.sort(sims)[-min(5, len(sims)):].mean()

    for name, vals in [
        ("avg_shape_conformity", avg_shape_sim),
        ("max_shape_conformity", max_shape_sim),
        ("top5_shape_conformity", top5_shape_sim),
    ]:
        r, p = spearmanr(vals, freshness)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        form_mean = vals[is_formulaic].mean()
        fresh_mean = vals[is_fresh].mean()
        print(f"  {name:25s}: r={r:+.4f} p={p:.6f} {sig:3s}  formulaic={form_mean:.4f} fresh={fresh_mean:.4f} gap={form_mean-fresh_mean:+.4f}")

    # ════════════════════════════════════════════════════════════
    # 3. RHYTHM-ONLY COMPARISON
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("3. RHYTHM-ONLY CONFORMITY (step size pattern only)")
    print("=" * 80)

    rhythm_matrix = np.array([s["shape_rhythm"] for s in shapes])
    rhythm_norms = np.linalg.norm(rhythm_matrix, axis=1, keepdims=True)
    rhythm_normed = rhythm_matrix / (rhythm_norms + 1e-10)
    rhythm_sim_matrix = rhythm_normed @ rhythm_normed.T

    avg_rhythm_sim = np.zeros(n)
    max_rhythm_sim = np.zeros(n)
    top5_rhythm_sim = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        sims = rhythm_sim_matrix[i, mask]
        avg_rhythm_sim[i] = sims.mean()
        max_rhythm_sim[i] = sims.max()
        top5_rhythm_sim[i] = np.sort(sims)[-min(5, len(sims)):].mean()

    for name, vals in [
        ("avg_rhythm_conformity", avg_rhythm_sim),
        ("max_rhythm_conformity", max_rhythm_sim),
        ("top5_rhythm_conformity", top5_rhythm_sim),
    ]:
        r, p = spearmanr(vals, freshness)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        form_mean = vals[is_formulaic].mean()
        fresh_mean = vals[is_fresh].mean()
        print(f"  {name:25s}: r={r:+.4f} p={p:.6f} {sig:3s}  formulaic={form_mean:.4f} fresh={fresh_mean:.4f} gap={form_mean-fresh_mean:+.4f}")

    # ════════════════════════════════════════════════════════════
    # 4. WEIGHT OPTIMIZATION
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("4. COMPOSITE SCORE OPTIMIZATION")
    print("=" * 80)

    all_features = {
        "max_jump_position": scalar_features["max_jump_position"],
        "final_ratio": scalar_features["final_ratio"],
        "step_entropy": scalar_features["step_entropy"],
        "step_autocorrelation": scalar_features["step_autocorrelation"],
        "avg_shape_conformity": avg_shape_sim,
        "max_shape_conformity": max_shape_sim,
        "top5_shape_conformity": top5_shape_sim,
        "avg_rhythm_conformity": avg_rhythm_sim,
        "max_rhythm_conformity": max_rhythm_sim,
        "top5_rhythm_conformity": top5_rhythm_sim,
    }

    # Show all correlations
    print("\n  All feature correlations with freshness:")
    ranked = []
    for name, vals in all_features.items():
        r, p = spearmanr(vals, freshness)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ranked.append((name, r, p, sig, vals))
        print(f"    {name:25s}: r={r:+.4f} p={p:.6f} {sig}")

    # Sort by absolute correlation
    ranked.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  Top features by |correlation|:")
    for name, r, p, sig, _ in ranked[:6]:
        print(f"    {name:25s}: r={r:+.4f} {sig}")

    # Use top 4 features for composite
    top_features = ranked[:4]
    print(f"\n  Using top 4 features for composite score:")

    # Normalize
    feat_arrays = []
    feat_names = []
    norm_params = {}
    for name, r, p, sig, vals in top_features:
        vmin, vmax = float(vals.min()), float(vals.max())
        norm_params[name] = {"min": vmin, "max": vmax}
        # If negative correlation with freshness, we want HIGH values = formulaic
        # So flip if r > 0 (positive r means high value = fresh, we want high = formulaic)
        if r > 0:
            normed = 1.0 - np.clip((vals - vmin) / (vmax - vmin + 1e-10), 0, 1)
            norm_params[name]["flip"] = True
        else:
            normed = np.clip((vals - vmin) / (vmax - vmin + 1e-10), 0, 1)
            norm_params[name]["flip"] = False
        feat_arrays.append(normed)
        feat_names.append(name)
        print(f"    {name:25s}: r={r:+.4f} {'(flipped)' if r > 0 else ''}")

    feature_matrix = np.column_stack(feat_arrays)

    # Grid search
    print("\n  Grid searching weights...")
    best_f1 = 0
    best_weights = None
    best_threshold = 0
    n_features = len(feat_names)

    weight_options = np.linspace(0, 1, 11)
    count = 0
    for w_combo in np.array(np.meshgrid(*[weight_options]*n_features)).T.reshape(-1, n_features):
        total = w_combo.sum()
        if total < 0.01:
            continue
        w = w_combo / total
        scores = feature_matrix @ w
        count += 1

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

    print(f"  Searched {count} weight combos")
    print(f"\n  *** OPTIMAL WEIGHTS ***")
    for name, w in zip(feat_names, best_weights):
        print(f"    w_{name} = {w:.4f}")

    final_scores = feature_matrix @ best_weights
    pred = final_scores >= best_threshold
    tp = (pred & is_formulaic).sum()
    fp = (pred & ~is_formulaic).sum()
    fn = (~pred & is_formulaic).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0

    comp_r, comp_p = spearmanr(final_scores, freshness)
    print(f"\n  Threshold: {best_threshold:.4f}")
    print(f"  F1={best_f1:.3f} P={prec:.3f} R={rec:.3f} TP={tp} FP={fp} FN={fn}")
    print(f"  Correlation with freshness: r={comp_r:+.4f} p={comp_p:.8f}")

    # Operating points
    print(f"\n  Operating points:")
    print(f"  {'Threshold':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    for pct in [40, 50, 60, 70, 75, 80, 85, 90, 95]:
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

    print("\n  MOST FORMULAIC paths (highest conformity):")
    for rank, i in enumerate(sorted_idx[:15]):
        oi = valid[i]
        o = openers[oi]
        s = shapes[i]
        print(f"    {rank+1:2d}. score={final_scores[i]:.3f} fresh={o['freshness']} maxjump@{s['max_jump_pos']:.2f} final_ratio={s['final_ratio']:.2f} entropy={s['step_entropy']:.2f}")
        print(f"        \"{o['text'][:85]}\"")

    print("\n  FRESHEST paths (lowest conformity):")
    for rank, i in enumerate(sorted_idx[-10:]):
        oi = valid[i]
        o = openers[oi]
        s = shapes[i]
        print(f"    {rank+1:2d}. score={final_scores[i]:.3f} fresh={o['freshness']} maxjump@{s['max_jump_pos']:.2f} final_ratio={s['final_ratio']:.2f}")
        print(f"        \"{o['text'][:85]}\"")

    # Score distribution
    print(f"\n  Score distribution by freshness:")
    for fv in sorted(set(freshness)):
        mask = freshness == fv
        if mask.sum() < 2:
            continue
        vals = final_scores[mask]
        print(f"    freshness={int(fv):2d}: n={mask.sum():4d}  mean={vals.mean():.3f}  std={vals.std():.3f}")

    # ════════════════════════════════════════════════════════════
    # FINAL
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("FINAL: PATH SHAPE FORMULA")
    print("=" * 80)
    print(f"""
  For each opener:
  1. Embed each word, compute step sizes between consecutive words
  2. Normalize step sizes to a rhythm vector (interpolated to {SHAPE_DIM}d)
  3. Compute: max_jump_position, final_ratio, step_entropy, step_autocorrelation
  4. Compute shape conformity: cosine sim of shape vector to all other openers

  Features used (top 4 by |correlation with freshness|):""")
    for name, w in zip(feat_names, best_weights):
        print(f"    {w:.3f} × {name}")
    print(f"""
  Threshold: {best_threshold:.4f}
  F1={best_f1:.3f}  P={prec:.3f}  R={rec:.3f}
  Correlation with freshness: r={comp_r:+.4f}

  NO template library. NO string matching. NO known patterns.
  Pure path shape geometry.
""")

    # Save
    model_data = {
        "version": 3,
        "type": "path_shape",
        "description": "Pure path shape detector. No templates, no string matching.",
        "embedding_model": EMBEDDING_MODEL,
        "shape_dim": SHAPE_DIM,
        "feature_names": feat_names,
        "weights": best_weights.tolist(),
        "normalization": norm_params,
        "threshold": float(best_threshold),
        "metrics": {"f1": float(best_f1), "precision": float(prec), "recall": float(rec)},
        "correlation": float(comp_r),
    }
    model_path = SRC_DIR / "path_shape_model.json"
    with open(model_path, "w") as f:
        json.dump(model_data, f, indent=2)
    print(f"  Saved to: {model_path}")


if __name__ == "__main__":
    main()
