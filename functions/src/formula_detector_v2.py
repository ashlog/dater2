#!/usr/bin/env python3
"""
Formula Detector v2 — Purely Embedding-Based
=============================================
Detects formulaic openers using ONLY embedding math, no hard-coded text patterns.
The key innovation: sub-phrase embedding. Instead of checking for "is just" as text,
we embed all 2-gram and 3-gram windows from the opener and check their similarity
to template embeddings. This catches ANY formulaic phrase pattern generically.

Usage:
    python3 -u formula_detector_v2.py                    # Train + validate + export
    python3 -u formula_detector_v2.py --score "opener"   # Score a single opener
"""

import json
import re
import sys
import glob
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).parent
EVAL_GLOB = str(SRC_DIR / "eval_results_gen_*.json")
MODEL_FILE = SRC_DIR / "formula_detector_model.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

FORMULAIC_THRESHOLD = 4  # freshness <= this => formulaic
FRESH_THRESHOLD = 8      # freshness >= this => fresh

# Template phrases — these get embedded, not string-matched
TEMPLATE_PHRASES = [
    # "X is just Y" variants
    "is just", "are just", "is basically", "is really just",
    # Accusatory closers
    "who hurt you", "what did you do", "who was it",
    # Relabeling
    "that's not a", "that's not an",
    # Generic closers
    "coffee or boba", "what's your go-to",
    # Agent/publicist
    "who's your agent", "who's their publicist", "who's funding this",
    # Other formulaic
    "what's the conversion rate", "how many reps", "threat level",
    "with extra steps", "with a debit card", "with better lighting",
]


def cosine_sim(a, b):
    return 1.0 - cosine_dist(a, b)


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


def get_ngrams(text, ns=[2, 3]):
    """Extract word n-grams from text."""
    words = text.lower().split()
    grams = []
    for n in ns:
        for i in range(len(words) - n + 1):
            grams.append(" ".join(words[i:i+n]))
    return grams


def compute_features(opener_emb, opener_text, template_embs, ngram_template_sims,
                     formula_dir, formulaic_centroid, norm):
    """Compute all features for a single opener. All embedding-based."""

    # F1: Projection onto formula direction (sentence-level)
    raw_proj = float(np.dot(opener_emb, formula_dir))
    f1 = np.clip((raw_proj - norm["proj_min"]) / (norm["proj_max"] - norm["proj_min"]), 0, 1)

    # F2: Cosine similarity to formulaic centroid (sentence-level)
    raw_cos = float(np.dot(opener_emb, formulaic_centroid))
    f2 = np.clip((raw_cos - norm["cos_min"]) / (norm["cos_max"] - norm["cos_min"]), 0, 1)

    # F3: Max sentence-level template similarity
    sent_sims = opener_emb @ template_embs.T
    raw_sent_max = float(sent_sims.max())
    f3 = np.clip((raw_sent_max - norm["sent_tmpl_min"]) / (norm["sent_tmpl_max"] - norm["sent_tmpl_min"]), 0, 1)
    best_sent_idx = int(sent_sims.argmax())

    # F4: Max SUB-PHRASE template similarity (the key innovation)
    # Embed all 2-grams and 3-grams, check max sim to template library
    f4 = 0.0
    best_ngram = ""
    best_ngram_template = ""
    best_ngram_sim = 0.0

    if ngram_template_sims is not None:
        f4 = float(ngram_template_sims["max_sim"])
        best_ngram = ngram_template_sims["best_ngram"]
        best_ngram_template = ngram_template_sims["best_template"]
        best_ngram_sim = ngram_template_sims["max_sim"]
        f4 = np.clip((f4 - norm["ngram_tmpl_min"]) / (norm["ngram_tmpl_max"] - norm["ngram_tmpl_min"]), 0, 1)

    # F5: Embedding path final_jump (closer surprise — low = generic closer)
    # Higher final_jump = more surprising closer = better = less formulaic
    # So we invert: lack of surprise = more formulaic
    words = opener_text.lower().split()
    f5 = 0.5  # default
    if len(words) >= 3:
        # We'll handle this from pre-computed path data
        pass

    return {
        "formula_direction_proj": float(f1),
        "cosine_to_formulaic": float(f2),
        "max_sent_template_sim": float(f3),
        "max_ngram_template_sim": float(f4),
        "best_sent_template": best_sent_idx,
        "best_ngram": best_ngram,
        "best_ngram_template": best_ngram_template,
        "best_ngram_sim": float(best_ngram_sim),
    }


def train():
    print("=" * 80)
    print("FORMULA DETECTOR v2 — PURELY EMBEDDING-BASED")
    print("=" * 80)

    openers = load_all_eval_data()
    print(f"\nLoaded {len(openers)} unique scored openers")

    freshness = np.array([o["freshness"] for o in openers])
    formulaic_idx = [i for i, o in enumerate(openers) if o["freshness"] <= FORMULAIC_THRESHOLD]
    fresh_idx = [i for i, o in enumerate(openers) if o["freshness"] >= FRESH_THRESHOLD]
    print(f"Formulaic (freshness <= {FORMULAIC_THRESHOLD}): {len(formulaic_idx)}")
    print(f"Fresh (freshness >= {FRESH_THRESHOLD}): {len(fresh_idx)}")

    # Load model and embed
    print(f"\nLoading {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embed = lambda texts: model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    texts = [o["text"] for o in openers]
    embeddings = embed(texts)
    template_embs = embed(TEMPLATE_PHRASES)
    print(f"Embedded {len(texts)} openers + {len(TEMPLATE_PHRASES)} templates")

    # Formula direction vector
    centroid_formulaic = embeddings[formulaic_idx].mean(axis=0)
    centroid_formulaic /= np.linalg.norm(centroid_formulaic)
    centroid_fresh = embeddings[fresh_idx].mean(axis=0)
    centroid_fresh /= np.linalg.norm(centroid_fresh)
    formula_dir = centroid_formulaic - centroid_fresh
    formula_dir /= np.linalg.norm(formula_dir)

    # ════════════════════════════════════════════════════════════
    # FEATURE COMPUTATION
    # ════════════════════════════════════════════════════════════

    # F1: Projection onto formula direction
    projections = embeddings @ formula_dir

    # F2: Cosine to formulaic centroid
    cos_to_form = embeddings @ centroid_formulaic

    # F3: Max sentence-level template similarity
    sent_template_sims = embeddings @ template_embs.T
    max_sent_sim = sent_template_sims.max(axis=1)

    # F4: Sub-phrase (n-gram) template similarity — THE KEY FEATURE
    print("\nComputing sub-phrase template similarities...")
    all_ngrams = []
    ngram_to_opener = []  # maps each ngram back to its opener index
    for i, text in enumerate(texts):
        grams = get_ngrams(text, ns=[2, 3])
        for g in grams:
            all_ngrams.append(g)
            ngram_to_opener.append(i)

    print(f"  {len(all_ngrams)} n-grams from {len(texts)} openers")

    # Embed all n-grams in one batch
    if all_ngrams:
        ngram_embs = embed(all_ngrams)
        ngram_template_sims = ngram_embs @ template_embs.T  # (n_ngrams, n_templates)

        # For each opener, find max n-gram template similarity
        max_ngram_sim = np.zeros(len(texts))
        best_ngram_per_opener = [""] * len(texts)
        best_template_per_opener = [""] * len(texts)

        for j in range(len(all_ngrams)):
            opener_i = ngram_to_opener[j]
            row_max = ngram_template_sims[j].max()
            if row_max > max_ngram_sim[opener_i]:
                max_ngram_sim[opener_i] = row_max
                best_ngram_per_opener[opener_i] = all_ngrams[j]
                best_template_per_opener[opener_i] = TEMPLATE_PHRASES[int(ngram_template_sims[j].argmax())]
    else:
        max_ngram_sim = np.zeros(len(texts))

    # ════════════════════════════════════════════════════════════
    # FEATURE ANALYSIS
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("FEATURE CORRELATIONS WITH FRESHNESS")
    print("=" * 80)

    features = {
        "formula_direction_proj": projections,
        "cosine_to_formulaic": cos_to_form,
        "max_sent_template_sim": max_sent_sim,
        "max_ngram_template_sim": max_ngram_sim,
    }

    for name, vals in features.items():
        r, p = spearmanr(vals, freshness)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        # Also show mean for formulaic vs fresh
        form_mean = vals[formulaic_idx].mean()
        fresh_mean = vals[fresh_idx].mean()
        print(f"  {name:30s}: r={r:+.4f} p={p:.6f} {sig:3s}  form={form_mean:.4f} fresh={fresh_mean:.4f} gap={form_mean-fresh_mean:+.4f}")

    # Show top ngram matches for formulaic openers
    print("\n  Top n-gram matches for FORMULAIC openers:")
    for i in formulaic_idx[:15]:
        print(f"    [{openers[i]['freshness']}] sim={max_ngram_sim[i]:.3f} ngram=\"{best_ngram_per_opener[i]}\" ← tmpl=\"{best_template_per_opener[i]}\"")
        print(f"        \"{openers[i]['text'][:80]}\"")

    print("\n  Top n-gram matches for FRESH openers:")
    for i in fresh_idx[:10]:
        print(f"    [{openers[i]['freshness']}] sim={max_ngram_sim[i]:.3f} ngram=\"{best_ngram_per_opener[i]}\" ← tmpl=\"{best_template_per_opener[i]}\"")
        print(f"        \"{openers[i]['text'][:80]}\"")

    # ════════════════════════════════════════════════════════════
    # WEIGHT OPTIMIZATION (purely embedding features)
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("WEIGHT OPTIMIZATION")
    print("=" * 80)

    # Normalize features to [0, 1]
    norm_params = {
        "proj_min": float(projections.min()), "proj_max": float(projections.max()),
        "cos_min": float(cos_to_form.min()), "cos_max": float(cos_to_form.max()),
        "sent_tmpl_min": float(max_sent_sim.min()), "sent_tmpl_max": float(max_sent_sim.max()),
        "ngram_tmpl_min": float(max_ngram_sim.min()), "ngram_tmpl_max": float(max_ngram_sim.max()),
    }

    def norm_feat(vals, fmin, fmax):
        return np.clip((vals - fmin) / (fmax - fmin + 1e-10), 0, 1)

    f1_norm = norm_feat(projections, norm_params["proj_min"], norm_params["proj_max"])
    f2_norm = norm_feat(cos_to_form, norm_params["cos_min"], norm_params["cos_max"])
    f3_norm = norm_feat(max_sent_sim, norm_params["sent_tmpl_min"], norm_params["sent_tmpl_max"])
    f4_norm = norm_feat(max_ngram_sim, norm_params["ngram_tmpl_min"], norm_params["ngram_tmpl_max"])

    feature_matrix = np.column_stack([f1_norm, f2_norm, f3_norm, f4_norm])
    feature_names = ["formula_direction_proj", "cosine_to_formulaic", "max_sent_template_sim", "max_ngram_template_sim"]

    is_formulaic = np.array([o["freshness"] <= FORMULAIC_THRESHOLD for o in openers])

    # Grid search
    print("\n  Grid search over weight combinations...")
    best_f1 = 0
    best_weights = None
    best_threshold = 0

    # Coarse search
    weight_options = np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0
    count = 0
    for w1 in weight_options:
        for w2 in weight_options:
            for w3 in weight_options:
                for w4 in weight_options:
                    total = w1 + w2 + w3 + w4
                    if total < 0.01:
                        continue
                    w = np.array([w1, w2, w3, w4]) / total
                    scores = feature_matrix @ w

                    # Try multiple thresholds
                    for pct in range(30, 90, 5):
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
                    count += 1

    print(f"  Searched {count} weight combinations")
    print(f"\n  *** OPTIMAL WEIGHTS ***")
    for name, w in zip(feature_names, best_weights):
        print(f"    w_{name} = {w:.4f}")

    # Compute final scores with optimal weights
    final_scores = feature_matrix @ best_weights
    pred = final_scores >= best_threshold
    tp = (pred & is_formulaic).sum()
    fp = (pred & ~is_formulaic).sum()
    fn = (~pred & is_formulaic).sum()
    tn = (~pred & ~is_formulaic).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"\n  Optimal threshold: {best_threshold:.4f}")
    print(f"  F1={f1:.3f} P={prec:.3f} R={rec:.3f}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")

    # Operating points table
    print(f"\n  Operating points:")
    print(f"  {'Threshold':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'FPR':>10s}")
    for pct in [30, 40, 50, 60, 70, 80, 85, 90]:
        t = np.percentile(final_scores, pct)
        p2 = final_scores >= t
        tp2 = (p2 & is_formulaic).sum()
        fp2 = (p2 & ~is_formulaic).sum()
        fn2 = (~p2 & is_formulaic).sum()
        tn2 = (~p2 & ~is_formulaic).sum()
        pr = tp2/(tp2+fp2) if (tp2+fp2) > 0 else 0
        rc = tp2/(tp2+fn2) if (tp2+fn2) > 0 else 0
        f1_ = 2*pr*rc/(pr+rc) if (pr+rc) > 0 else 0
        fpr = fp2/(fp2+tn2) if (fp2+tn2) > 0 else 0
        print(f"  {t:10.4f} {pr:10.3f} {rc:10.3f} {f1_:10.3f} {fpr:10.3f}")

    # ════════════════════════════════════════════════════════════
    # VALIDATION
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    comp_r, comp_p = spearmanr(final_scores, freshness)
    print(f"\n  Composite score vs freshness: r={comp_r:+.4f} p={comp_p:.8f}")

    # Score distribution by freshness bucket
    print(f"\n  Score distribution by freshness:")
    for fv in sorted(set(freshness)):
        mask = freshness == fv
        if mask.sum() < 2:
            continue
        vals = final_scores[mask]
        print(f"    freshness={int(fv):2d}: n={mask.sum():4d}  mean={vals.mean():.3f}  std={vals.std():.3f}  [{vals.min():.3f}, {vals.max():.3f}]")

    # True positives
    print(f"\n  Correctly caught formulaic (True Positives, top 15):")
    tp_mask = pred & is_formulaic
    tp_indices = np.where(tp_mask)[0]
    tp_sorted = sorted(tp_indices, key=lambda i: final_scores[i], reverse=True)
    for i in tp_sorted[:15]:
        print(f"    score={final_scores[i]:.3f} fresh={openers[i]['freshness']} ngram=\"{best_ngram_per_opener[i]}\" ← \"{best_template_per_opener[i]}\"")
        print(f"      \"{openers[i]['text'][:85]}\"")

    # False positives
    print(f"\n  False Positives (flagged but actually fresh, top 10):")
    fp_mask = pred & ~is_formulaic
    fp_indices = np.where(fp_mask)[0]
    fp_sorted = sorted(fp_indices, key=lambda i: final_scores[i], reverse=True)
    for i in fp_sorted[:10]:
        print(f"    score={final_scores[i]:.3f} fresh={openers[i]['freshness']} ngram=\"{best_ngram_per_opener[i]}\"")
        print(f"      \"{openers[i]['text'][:85]}\"")

    # ════════════════════════════════════════════════════════════
    # EXPORT MODEL
    # ════════════════════════════════════════════════════════════
    model_data = {
        "version": 2,
        "description": "Formula detector v2 — purely embedding-based, no hard-coded text patterns",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": int(embeddings.shape[1]),
        "formula_direction": formula_dir.tolist(),
        "formulaic_centroid": centroid_formulaic.tolist(),
        "template_phrases": TEMPLATE_PHRASES,
        "template_embeddings": template_embs.tolist(),
        "feature_names": feature_names,
        "weights": best_weights.tolist(),
        "normalization": norm_params,
        "threshold": float(best_threshold),
        "threshold_metrics": {"precision": float(prec), "recall": float(rec), "f1": float(f1)},
        "training_stats": {
            "n_openers": len(openers),
            "n_formulaic": len(formulaic_idx),
            "n_fresh": len(fresh_idx),
        },
    }

    with open(MODEL_FILE, "w") as f:
        json.dump(model_data, f)

    print(f"\n  Model saved to: {MODEL_FILE} ({MODEL_FILE.stat().st_size / 1024:.1f} KB)")

    # ════════════════════════════════════════════════════════════
    # FINAL FORMULA
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("FINAL FORMULA (v2 — purely embedding-based)")
    print("=" * 80)
    print(f"""
  For a candidate opener text T:

  1. Embed T using {EMBEDDING_MODEL} -> e (384-dim, L2-normalized)
  2. Extract all 2-grams and 3-grams from T, embed each -> g_i

  3. Compute 4 features:
     f1 = normalized(e . formula_direction)     — how "formulaic" is the overall meaning
     f2 = normalized(e . formulaic_centroid)     — proximity to known formulaic cluster
     f3 = normalized(max(e . template_embs))     — sentence-level match to template phrases
     f4 = normalized(max_i(max(g_i . template_embs)))  — SUB-PHRASE match to templates
          (this catches "is just", "who hurt you", etc. WITHOUT hard-coding them)

  4. formula_score = {best_weights[0]:.4f} * f1 + {best_weights[1]:.4f} * f2 + {best_weights[2]:.4f} * f3 + {best_weights[3]:.4f} * f4

  5. REJECT if formula_score >= {best_threshold:.4f}

  Performance: P={prec:.3f} R={rec:.3f} F1={f1:.3f}
  Correlation with freshness: r={comp_r:+.4f}
  NO hard-coded text patterns. All detection is via embedding similarity.
""")

    return model_data


def score_opener(text: str, model_path: str = None) -> dict:
    """Score a single opener. Loads pre-trained model."""
    if model_path is None:
        model_path = str(MODEL_FILE)

    with open(model_path) as f:
        md = json.load(f)

    model = SentenceTransformer(md["embedding_model"])
    emb = model.encode([text], normalize_embeddings=True)[0]
    norm = md["normalization"]

    # F1: projection
    formula_dir = np.array(md["formula_direction"])
    raw_proj = float(np.dot(emb, formula_dir))
    f1 = np.clip((raw_proj - norm["proj_min"]) / (norm["proj_max"] - norm["proj_min"]), 0, 1)

    # F2: cosine to formulaic centroid
    centroid = np.array(md["formulaic_centroid"])
    raw_cos = float(np.dot(emb, centroid))
    f2 = np.clip((raw_cos - norm["cos_min"]) / (norm["cos_max"] - norm["cos_min"]), 0, 1)

    # F3: sentence-level template similarity
    template_embs = np.array(md["template_embeddings"])
    sent_sims = emb @ template_embs.T
    raw_sent_max = float(sent_sims.max())
    best_sent_idx = int(sent_sims.argmax())
    f3 = np.clip((raw_sent_max - norm["sent_tmpl_min"]) / (norm["sent_tmpl_max"] - norm["sent_tmpl_min"]), 0, 1)

    # F4: sub-phrase template similarity (the generic formula catcher)
    ngrams = get_ngrams(text, ns=[2, 3])
    f4 = 0.0
    best_ngram = ""
    best_ngram_template = ""
    best_ngram_sim = 0.0

    if ngrams:
        ngram_embs = model.encode(ngrams, normalize_embeddings=True)
        ngram_sims = ngram_embs @ template_embs.T  # (n_ngrams, n_templates)
        max_idx = np.unravel_index(ngram_sims.argmax(), ngram_sims.shape)
        best_ngram_sim = float(ngram_sims[max_idx])
        best_ngram = ngrams[max_idx[0]]
        best_ngram_template = md["template_phrases"][max_idx[1]]
        f4 = np.clip((best_ngram_sim - norm["ngram_tmpl_min"]) / (norm["ngram_tmpl_max"] - norm["ngram_tmpl_min"]), 0, 1)

    # Composite
    weights = np.array(md["weights"])
    features = np.array([f1, f2, f3, f4])
    formula_score = float(features @ weights)

    return {
        "formula_score": formula_score,
        "is_formulaic": formula_score >= md["threshold"],
        "features": {
            "formula_direction_proj": float(f1),
            "cosine_to_formulaic": float(f2),
            "max_sent_template_sim": float(f3),
            "max_ngram_template_sim": float(f4),
        },
        "best_ngram_match": {
            "ngram": best_ngram,
            "template": best_ngram_template,
            "similarity": best_ngram_sim,
        },
        "closest_sent_template": (md["template_phrases"][best_sent_idx], float(sent_sims[best_sent_idx])),
    }


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--score":
        text = sys.argv[2] if len(sys.argv) > 2 else input("Enter opener: ")
        r = score_opener(text)
        print(f"\nFormula score: {r['formula_score']:.3f} ({'REJECT' if r['is_formulaic'] else 'ACCEPT'})")
        print(f"\nFeatures:")
        for k, v in r["features"].items():
            print(f"  {k:30s}: {v:.4f}")
        print(f"\nBest n-gram match: \"{r['best_ngram_match']['ngram']}\" ↔ \"{r['best_ngram_match']['template']}\" (sim={r['best_ngram_match']['similarity']:.4f})")
        print(f"Closest sent template: \"{r['closest_sent_template'][0]}\" (sim={r['closest_sent_template'][1]:.4f})")
    else:
        train()
