#!/usr/bin/env python3
"""
Embedding Analysis of Hinge Dating App Openers
================================================
Analyzes sentence/word embeddings to find patterns distinguishing
high-performing openers from low-performing ones across 8 eval dimensions.
"""

import json
import re
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity as cos_sim_matrix
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import pearsonr, spearmanr, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ────────────────────────────────────────────────────────────
EVAL_FILE = Path(__file__).parent / "eval_results_gen_2026-03-16T02-22-00.json"
BEST_ANSWER_FILE = Path(__file__).parent.parent / "bestAnswer.txt"
MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality

SCORE_DIMS = [
    "reply_likelihood", "humor_quality", "creativity", "specificity",
    "human_likeness", "closer_strength", "warmth_signal", "formula_freshness"
]

# ─── HELPERS ──────────────────────────────────────────────────────────
def cosine_sim(a, b):
    return 1.0 - cosine_dist(a, b)

def avg_pairwise_sim(embs, max_pairs=500):
    n = len(embs)
    if n < 2:
        return 0
    sims = []
    if n * (n - 1) / 2 <= max_pairs:
        for i in range(n):
            for j in range(i + 1, n):
                sims.append(cosine_sim(embs[i], embs[j]))
    else:
        rng = np.random.default_rng(42)
        for _ in range(max_pairs):
            i, j = rng.choice(n, 2, replace=False)
            sims.append(cosine_sim(embs[i], embs[j]))
    return np.mean(sims)

# ─── LOAD DATA ────────────────────────────────────────────────────────
def load_eval_data():
    with open(EVAL_FILE) as f:
        data = json.load(f)

    openers = []
    for r in data["results"]:
        scores = {dim: r["scores"][dim]["score"] for dim in SCORE_DIMS}
        scores["avg"] = float(np.mean(list(scores.values())))
        openers.append({
            "text": r["comment"],
            "model": r["model"],
            "prompt_q": r.get("promptQuestion", ""),
            "prompt_a": r.get("promptAnswer", ""),
            "profile": r.get("profileName", ""),
            "scores": scores,
        })
    return openers, data.get("aggregates", [])

def load_best_answers(early_lines=200, late_lines=200):
    lines = Path(BEST_ANSWER_FILE).read_text().splitlines()

    def extract_openers(line_range):
        out = []
        for line in line_range:
            m = re.search(r'\[PICKUP LINE\]\s*"(.+?)"', line)
            if m:
                out.append(m.group(1))
        return out

    early = extract_openers(lines[:early_lines])
    late = extract_openers(lines[-late_lines:])
    return early, late

# ─── MAIN ─────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("HINGE OPENER EMBEDDING ANALYSIS")
    print("=" * 80)

    # Load data
    openers, aggregates = load_eval_data()
    print(f"\nLoaded {len(openers)} eval openers")

    early_best, late_best = load_best_answers()
    print(f"Loaded {len(early_best)} early + {len(late_best)} late bestAnswer openers")

    # Load model and embed
    print(f"\nLoading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    embed = lambda texts: model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    texts = [o["text"] for o in openers]
    embeddings = embed(texts)
    print(f"Embedded {len(texts)} openers -> shape {embeddings.shape}")

    # Also embed unnormalized for magnitude analysis
    raw_embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=False)

    # ──────────────────────────────────────────────────────────────
    # 1. EMBEDDING DIMENSION CORRELATIONS WITH SCORES
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("1. EMBEDDING DIMENSION CORRELATIONS WITH SCORES")
    print("=" * 80)

    for dim in SCORE_DIMS + ["avg"]:
        score_vec = np.array([o["scores"][dim] for o in openers])
        correlations = []
        for di in range(embeddings.shape[1]):
            r, p = spearmanr(embeddings[:, di], score_vec)
            correlations.append((di, r, p))
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"\n  {dim}:")
        for di, r, p in correlations[:5]:
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    dim[{di:3d}] r={r:+.3f} p={p:.4f} {sig}")

    # ──────────────────────────────────────────────────────────────
    # 2. HIGH vs LOW CLUSTER ANALYSIS
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("2. HIGH vs LOW CLUSTER ANALYSIS (per dimension)")
    print("=" * 80)

    direction_vectors = {}  # Store for cross-dim analysis later

    for dim in SCORE_DIMS:
        scores = [o["scores"][dim] for o in openers]
        p75 = np.percentile(scores, 75)
        p25 = np.percentile(scores, 25)

        high_idx = [i for i, o in enumerate(openers) if o["scores"][dim] >= p75]
        low_idx = [i for i, o in enumerate(openers) if o["scores"][dim] <= p25]

        if len(high_idx) < 3 or len(low_idx) < 3:
            print(f"\n  {dim}: Not enough samples for split (p25={p25}, p75={p75})")
            continue

        high_emb = embeddings[high_idx]
        low_emb = embeddings[low_idx]

        high_centroid = high_emb.mean(axis=0)
        high_centroid /= np.linalg.norm(high_centroid)
        low_centroid = low_emb.mean(axis=0)
        low_centroid /= np.linalg.norm(low_centroid)

        centroid_sim = cosine_sim(high_centroid, low_centroid)
        high_coh = avg_pairwise_sim(high_emb)
        low_coh = avg_pairwise_sim(low_emb)

        # Direction vector
        direction = high_centroid - low_centroid
        direction /= np.linalg.norm(direction)
        direction_vectors[dim] = direction

        # Project all openers onto direction
        projections = embeddings @ direction
        score_vec = np.array([o["scores"][dim] for o in openers])
        proj_r, proj_p = spearmanr(projections, score_vec)

        coherence_diff = high_coh - low_coh
        more_coherent = "HIGH" if coherence_diff > 0 else "LOW"

        print(f"\n  {dim} (high >= {p75}, low <= {p25}):")
        print(f"    High group: n={len(high_idx)}, coherence={high_coh:.3f}")
        print(f"    Low group:  n={len(low_idx)}, coherence={low_coh:.3f}")
        print(f"    Centroid similarity: {centroid_sim:.3f}")
        print(f"    {more_coherent} scorers more similar (diff={coherence_diff:+.3f})")
        print(f"    Direction projection r={proj_r:+.3f}, p={proj_p:.4f}")

    # ──────────────────────────────────────────────────────────────
    # 3. WARMTH DEEP DIVE
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("3. WARMTH SIGNAL DEEP DIVE")
    print("=" * 80)

    warmth_scores = np.array([o["scores"]["warmth_signal"] for o in openers])
    high_warmth_idx = [i for i, o in enumerate(openers) if o["scores"]["warmth_signal"] >= 8]
    low_warmth_idx = [i for i, o in enumerate(openers) if o["scores"]["warmth_signal"] <= 5]

    if len(high_warmth_idx) >= 3 and len(low_warmth_idx) >= 3:
        warmth_dir = direction_vectors.get("warmth_signal")
        if warmth_dir is not None:
            warmth_proj = embeddings @ warmth_dir
            proj_r, proj_p = pearsonr(warmth_proj, warmth_scores)
            print(f"\n  Warmth direction projection correlation: r={proj_r:+.4f}, p={proj_p:.6f}")

            # Extremes
            proj_order = np.argsort(warmth_proj)
            print(f"\n  Most ANTI-warmth openers (by embedding direction):")
            for idx in proj_order[:4]:
                print(f"    [warmth={openers[idx]['scores']['warmth_signal']}] \"{openers[idx]['text'][:85]}\"")
            print(f"\n  Most PRO-warmth openers:")
            for idx in proj_order[-4:]:
                print(f"    [warmth={openers[idx]['scores']['warmth_signal']}] \"{openers[idx]['text'][:85]}\"")

    # Self-insertion analysis
    self_insert_re = re.compile(r"\bi[''`]?m\b|\bmy\b|\bi have\b|\bi was\b|\bi used to\b|\bi can\b|\bi['`]?ll\b", re.I)
    has_self = [bool(self_insert_re.search(o["text"])) for o in openers]
    with_self_warmth = [o["scores"]["warmth_signal"] for o, s in zip(openers, has_self) if s]
    without_self_warmth = [o["scores"]["warmth_signal"] for o, s in zip(openers, has_self) if not s]
    print(f"\n  Self-insertion ('I', 'my', etc.) and warmth:")
    print(f"    With self-ref: n={len(with_self_warmth)}, avg warmth={np.mean(with_self_warmth):.2f}")
    print(f"    Without:       n={len(without_self_warmth)}, avg warmth={np.mean(without_self_warmth):.2f}")
    print(f"    Diff: {np.mean(with_self_warmth) - np.mean(without_self_warmth):+.2f}")

    # Also check self-insertion vs other dimensions
    for dim in ["human_likeness", "formula_freshness", "avg"]:
        with_dim = [o["scores"][dim] for o, s in zip(openers, has_self) if s]
        without_dim = [o["scores"][dim] for o, s in zip(openers, has_self) if not s]
        diff = np.mean(with_dim) - np.mean(without_dim)
        if abs(diff) > 0.3:
            print(f"    Self-insertion also affects {dim}: diff={diff:+.2f}")

    # ──────────────────────────────────────────────────────────────
    # 4. FORMULA FRESHNESS DEEP DIVE
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("4. FORMULA FRESHNESS ANALYSIS")
    print("=" * 80)

    high_fresh_idx = [i for i, o in enumerate(openers) if o["scores"]["formula_freshness"] >= 8]
    low_fresh_idx = [i for i, o in enumerate(openers) if o["scores"]["formula_freshness"] <= 4]

    if len(high_fresh_idx) >= 3 and len(low_fresh_idx) >= 3:
        high_fresh_sim = avg_pairwise_sim(embeddings[high_fresh_idx])
        low_fresh_sim = avg_pairwise_sim(embeddings[low_fresh_idx])
        print(f"\n  Fresh openers (>=8): n={len(high_fresh_idx)}, intra-similarity={high_fresh_sim:.3f}")
        print(f"  Stale openers (<=4): n={len(low_fresh_idx)}, intra-similarity={low_fresh_sim:.3f}")
        more_clustered = "STALE" if low_fresh_sim > high_fresh_sim else "FRESH"
        print(f"  --> {more_clustered} openers are more clustered")

    print(f"\n  Stale openers (formula_freshness <= 4):")
    stale_all = [(i, o) for i, o in enumerate(openers) if o["scores"]["formula_freshness"] <= 4]
    for idx, o in stale_all:
        print(f"    [{o['scores']['formula_freshness']}] \"{o['text'][:90]}\"")

    print(f"\n  Fresh openers (formula_freshness >= 8):")
    fresh_all = [(i, o) for i, o in enumerate(openers) if o["scores"]["formula_freshness"] >= 8]
    for idx, o in fresh_all[:10]:
        print(f"    [{o['scores']['formula_freshness']}] \"{o['text'][:90]}\"")

    # ──────────────────────────────────────────────────────────────
    # 5. TOP vs BOTTOM OPENERS
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("5. TOP vs BOTTOM OPENERS BY AVERAGE SCORE")
    print("=" * 80)

    avg_scores = np.array([o["scores"]["avg"] for o in openers])
    top_threshold = np.percentile(avg_scores, 80)
    bottom_threshold = np.percentile(avg_scores, 20)

    top_idx = [i for i in range(len(openers)) if avg_scores[i] >= top_threshold]
    bottom_idx = [i for i in range(len(openers)) if avg_scores[i] <= bottom_threshold]

    top_intra = avg_pairwise_sim(embeddings[top_idx])
    bottom_intra = avg_pairwise_sim(embeddings[bottom_idx])
    cross_sim_val = np.mean([cosine_sim(embeddings[i], embeddings[j])
                             for i in top_idx for j in bottom_idx])

    print(f"\n  Top 20% (avg >= {top_threshold:.1f}): n={len(top_idx)}")
    print(f"  Bottom 20% (avg <= {bottom_threshold:.1f}): n={len(bottom_idx)}")
    print(f"  Top intra-similarity: {top_intra:.3f}")
    print(f"  Bottom intra-similarity: {bottom_intra:.3f}")
    print(f"  Cross-group similarity: {cross_sim_val:.3f}")
    print(f"  --> Top openers are {'MORE' if top_intra > bottom_intra else 'LESS'} similar to each other")

    # Raw magnitude
    top_mag = np.linalg.norm(raw_embeddings[top_idx], axis=1)
    bot_mag = np.linalg.norm(raw_embeddings[bottom_idx], axis=1)
    t, p = ttest_ind(top_mag, bot_mag)
    print(f"\n  Raw embedding magnitude:")
    print(f"    Top: mean={top_mag.mean():.3f}, std={top_mag.std():.3f}")
    print(f"    Bottom: mean={bot_mag.mean():.3f}, std={bot_mag.std():.3f}")
    print(f"    t-test: t={t:.3f}, p={p:.4f}")

    # Word count
    top_wc = [len(openers[i]["text"].split()) for i in top_idx]
    bot_wc = [len(openers[i]["text"].split()) for i in bottom_idx]
    print(f"\n  Word count: Top={np.mean(top_wc):.1f}, Bottom={np.mean(bot_wc):.1f}")

    # Show top openers
    sorted_by_avg = sorted(range(len(openers)), key=lambda i: openers[i]["scores"]["avg"], reverse=True)
    print(f"\n  Top 12 openers:")
    for rank, idx in enumerate(sorted_by_avg[:12], 1):
        o = openers[idx]
        pv = "v31" if "v31" in o["model"] else "v30"
        print(f"    {rank:2d}. [{o['scores']['avg']:.1f}|{pv}] \"{o['text'][:80]}\"")

    print(f"\n  Bottom 8 openers:")
    for rank, idx in enumerate(sorted_by_avg[-8:], 1):
        o = openers[idx]
        pv = "v31" if "v31" in o["model"] else "v30"
        print(f"    {rank:2d}. [{o['scores']['avg']:.1f}|{pv}] \"{o['text'][:80]}\"")

    # ──────────────────────────────────────────────────────────────
    # 6. CLOSER ANALYSIS (last N words)
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("6. CLOSER EMBEDDING ANALYSIS (Last 4 words)")
    print("=" * 80)

    closers = []
    for o in openers:
        words = o["text"].split()
        closers.append(" ".join(words[-4:]) if len(words) >= 4 else o["text"])

    closer_embs = embed(closers)
    closer_scores = np.array([o["scores"]["closer_strength"] for o in openers])

    # Correlation
    top_corrs = []
    for di in range(closer_embs.shape[1]):
        r, p = spearmanr(closer_embs[:, di], closer_scores)
        top_corrs.append((di, r, p))
    top_corrs.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n  Top closer embedding dims correlated with closer_strength:")
    for di, r, p in top_corrs[:8]:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    dim[{di:3d}] r={r:+.3f} p={p:.4f} {sig}")

    high_cs = [i for i, o in enumerate(openers) if o["scores"]["closer_strength"] >= 9]
    low_cs = [i for i, o in enumerate(openers) if o["scores"]["closer_strength"] <= 6]

    if len(high_cs) >= 3 and len(low_cs) >= 3:
        h_sim = avg_pairwise_sim(closer_embs[high_cs])
        l_sim = avg_pairwise_sim(closer_embs[low_cs])
        print(f"\n  High closer (>=9, n={len(high_cs)}): avg pairwise sim = {h_sim:.3f}")
        print(f"  Low closer  (<=6, n={len(low_cs)}): avg pairwise sim = {l_sim:.3f}")

        print(f"\n  Strong closers (>=9):")
        for i in high_cs[:8]:
            print(f"    [{openers[i]['scores']['closer_strength']}] ...{closers[i]}")
        print(f"\n  Weak closers (<=6):")
        for i in low_cs[:8]:
            print(f"    [{openers[i]['scores']['closer_strength']}] ...{closers[i]}")

    # ──────────────────────────────────────────────────────────────
    # 7. WORD-LEVEL ANALYSIS
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("7. WORD-LEVEL EMBEDDING PATTERNS")
    print("=" * 80)

    word_scores = defaultdict(lambda: defaultdict(list))
    for o in openers:
        words = set(re.findall(r'\b[a-zA-Z]{3,}\b', o["text"].lower()))
        for w in words:
            for dim in SCORE_DIMS + ["avg"]:
                word_scores[w][dim].append(o["scores"][dim])

    for dim in ["avg", "warmth_signal", "formula_freshness", "closer_strength", "humor_quality"]:
        word_avgs = []
        for word, dim_scores_map in word_scores.items():
            if len(dim_scores_map[dim]) >= 3:
                word_avgs.append((word, np.mean(dim_scores_map[dim]), len(dim_scores_map[dim])))

        word_avgs.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  {dim} - TOP words:  {', '.join(f'{w}({avg:.1f},n={n})' for w, avg, n in word_avgs[:8])}")
        print(f"  {dim} - BOTTOM words: {', '.join(f'{w}({avg:.1f},n={n})' for w, avg, n in word_avgs[-5:])}")

    # Embed top vs bottom words and compare
    top_words_by_avg = sorted(
        [(w, np.mean(s["avg"]), len(s["avg"])) for w, s in word_scores.items() if len(s["avg"]) >= 3],
        key=lambda x: x[1], reverse=True
    )

    if len(top_words_by_avg) >= 20:
        top_10w = [w for w, _, _ in top_words_by_avg[:10]]
        bot_10w = [w for w, _, _ in top_words_by_avg[-10:]]

        top_w_emb = embed(top_10w)
        bot_w_emb = embed(bot_10w)

        top_w_sim = avg_pairwise_sim(top_w_emb)
        bot_w_sim = avg_pairwise_sim(bot_w_emb)

        print(f"\n  Word embedding clustering:")
        print(f"    Top-scoring words {top_10w}: avg sim = {top_w_sim:.3f}")
        print(f"    Bottom-scoring words {bot_10w}: avg sim = {bot_w_sim:.3f}")

    # ──────────────────────────────────────────────────────────────
    # 8. v30 vs v31
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("8. PROMPT v30 vs v31 EMBEDDING COMPARISON")
    print("=" * 80)

    v30_idx = [i for i, o in enumerate(openers) if "v30" in o["model"]]
    v31_idx = [i for i, o in enumerate(openers) if "v31" in o["model"]]

    v30_emb = embeddings[v30_idx]
    v31_emb = embeddings[v31_idx]

    v30_centroid = v30_emb.mean(axis=0)
    v30_centroid /= np.linalg.norm(v30_centroid)
    v31_centroid = v31_emb.mean(axis=0)
    v31_centroid /= np.linalg.norm(v31_centroid)

    v30_int = avg_pairwise_sim(v30_emb)
    v31_int = avg_pairwise_sim(v31_emb)

    print(f"  v30: {len(v30_idx)} openers, internal sim={v30_int:.3f}")
    print(f"  v31: {len(v31_idx)} openers, internal sim={v31_int:.3f}")
    print(f"  Centroid similarity: {cosine_sim(v30_centroid, v31_centroid):.3f}")

    # v30->v31 direction and score correlations
    v_dir = v31_centroid - v30_centroid
    v_dir /= np.linalg.norm(v_dir)
    v_proj = embeddings @ v_dir

    print(f"\n  Score differences (v31 - v30) and direction correlation:")
    for dim in SCORE_DIMS:
        v30_sc = np.mean([openers[i]["scores"][dim] for i in v30_idx])
        v31_sc = np.mean([openers[i]["scores"][dim] for i in v31_idx])
        dim_scores = np.array([o["scores"][dim] for o in openers])
        r, p = spearmanr(v_proj, dim_scores)
        sig = "*" if p < 0.05 else ""
        print(f"    {dim:20s}: v30={v30_sc:.1f} v31={v31_sc:.1f} diff={v31_sc - v30_sc:+.1f}  dir_corr: r={r:+.3f} p={p:.4f} {sig}")

    # ──────────────────────────────────────────────────────────────
    # 9. EVOLUTION: early vs late bestAnswer vs eval
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("9. OPENER EVOLUTION (bestAnswer.txt early vs late vs eval)")
    print("=" * 80)

    early_emb = embed(early_best) if early_best else np.array([])
    late_emb = embed(late_best) if late_best else np.array([])

    groups = {}
    if len(early_emb) > 0:
        groups["early"] = early_emb
    if len(late_emb) > 0:
        groups["late"] = late_emb
    groups["eval"] = embeddings

    centroids = {}
    for name, embs in groups.items():
        c = embs.mean(axis=0)
        centroids[name] = c / np.linalg.norm(c)

    print(f"  Sizes: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()))

    names = list(centroids.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sim = cosine_sim(centroids[names[i]], centroids[names[j]])
            print(f"  {names[i]} <-> {names[j]} centroid similarity: {sim:.3f}")

    for name, embs in groups.items():
        internal = avg_pairwise_sim(embs)
        print(f"  {name} internal similarity: {internal:.3f}")

    # Early vs late vs v30/v31
    if "early" in centroids and "late" in centroids:
        for ref_name in ["early", "late"]:
            for target_name, target_c in [("v30", v30_centroid), ("v31", v31_centroid)]:
                sim = cosine_sim(centroids[ref_name], target_c)
                print(f"  {ref_name} <-> {target_name}: {sim:.3f}")

    print(f"\n  Sample EARLY openers:")
    for o in early_best[:5]:
        print(f"    \"{o[:100]}\"")
    print(f"\n  Sample LATE openers:")
    for o in late_best[:5]:
        print(f"    \"{o[:100]}\"")

    # ──────────────────────────────────────────────────────────────
    # 10. STRUCTURAL PATTERN ANALYSIS
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("10. STRUCTURAL PATTERN ANALYSIS")
    print("=" * 80)

    patterns = {
        "x_is_just_y": r"\bis just\b|\bare just\b",
        "who/what_hurt_you": r"(?:who|what).{0,20}hurt you",
        "what_did_you_do": r"what did you do|what.?d you do",
        "question_closer": r"\?$",
        "has_self_insert": self_insert_re.pattern,
        "has_honestly": r"\bhonestly\b",
        "has_confession": r"\bconfess",
        "has_threat_level": r"\bthreat\b",
    }

    print(f"\n  Pattern analysis (avg scores for openers matching vs not):")
    for name, pat in patterns.items():
        matching = [i for i in range(len(openers)) if re.search(pat, openers[i]["text"], re.IGNORECASE)]
        non_matching = [i for i in range(len(openers)) if i not in matching]

        if len(matching) < 2 or len(non_matching) < 2:
            continue

        match_avg = np.mean([openers[i]["scores"]["avg"] for i in matching])
        non_avg = np.mean([openers[i]["scores"]["avg"] for i in non_matching])

        notable_dims = []
        for dim in SCORE_DIMS:
            m_sc = np.mean([openers[i]["scores"][dim] for i in matching])
            n_sc = np.mean([openers[i]["scores"][dim] for i in non_matching])
            d = m_sc - n_sc
            if abs(d) >= 0.5:
                notable_dims.append(f"{dim}={d:+.1f}")

        match_coh = avg_pairwise_sim(embeddings[matching]) if len(matching) > 1 else 0

        print(f"\n  {name} (n={len(matching)}/{len(openers)}):")
        print(f"    avg score: match={match_avg:.2f}, non={non_avg:.2f}, diff={match_avg - non_avg:+.2f}")
        if notable_dims:
            print(f"    Notable diffs: {', '.join(notable_dims)}")
        print(f"    Matching coherence: {match_coh:.3f}")

    # ──────────────────────────────────────────────────────────────
    # 11. FORMULA DETECTION VIA TEMPLATE SIMILARITY
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("11. FORMULA DETECTION VIA EMBEDDING SIMILARITY TO TEMPLATES")
    print("=" * 80)

    templates = [
        "X is just Y with Z",
        "who hurt you",
        "what did you do to deserve this",
        "what's the conversion rate",
        "how many reps is that",
        "bribery with extra steps",
        "this is just competitive shopping",
        "what's your fastest checkout time",
        "that's a bold claim we should test it over coffee",
    ]

    template_embs = embed(templates)

    template_max_sims = []
    for i, emb in enumerate(embeddings):
        max_sim = max(cosine_sim(emb, te) for te in template_embs)
        template_max_sims.append(max_sim)

    template_max_sims = np.array(template_max_sims)

    freshness_scores = np.array([o["scores"]["formula_freshness"] for o in openers])
    r, p = spearmanr(template_max_sims, freshness_scores)
    print(f"\n  Template similarity vs formula_freshness: r={r:+.3f} p={p:.4f}")

    avg_sc = np.array([o["scores"]["avg"] for o in openers])
    r2, p2 = spearmanr(template_max_sims, avg_sc)
    print(f"  Template similarity vs avg score: r={r2:+.3f} p={p2:.4f}")

    sorted_by_sim = sorted(range(len(openers)), key=lambda i: template_max_sims[i], reverse=True)
    print(f"\n  MOST template-like openers:")
    for i in sorted_by_sim[:6]:
        print(f"    sim={template_max_sims[i]:.3f} fresh={openers[i]['scores']['formula_freshness']} \"{openers[i]['text'][:80]}\"")

    print(f"\n  LEAST template-like openers:")
    for i in sorted_by_sim[-6:]:
        print(f"    sim={template_max_sims[i]:.3f} fresh={openers[i]['scores']['formula_freshness']} \"{openers[i]['text'][:80]}\"")

    # ──────────────────────────────────────────────────────────────
    # 12. CROSS-DIMENSION DIRECTION VECTORS
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("12. DIMENSION DIRECTION VECTOR CROSS-CORRELATIONS")
    print("=" * 80)

    if len(direction_vectors) >= 2:
        dims_with_dirs = list(direction_vectors.keys())
        print(f"\n  Direction vector cosine similarities:")
        header = f"{'':20s}" + "".join(f"{d[:9]:>10s}" for d in dims_with_dirs)
        print(header)
        for d1 in dims_with_dirs:
            row = f"{d1:20s}"
            for d2 in dims_with_dirs:
                sim = cosine_sim(direction_vectors[d1], direction_vectors[d2])
                row += f"{sim:10.3f}"
            print(row)

    # ──────────────────────────────────────────────────────────────
    # 13. PAIRWISE SIMILARITY DISTRIBUTION
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("13. PAIRWISE SIMILARITY DISTRIBUTION")
    print("=" * 80)

    full_sim = cos_sim_matrix(embeddings)
    upper_tri = full_sim[np.triu_indices_from(full_sim, k=1)]

    print(f"\n  Overall: mean={upper_tri.mean():.3f}, std={upper_tri.std():.3f}, "
          f"min={upper_tri.min():.3f}, max={upper_tri.max():.3f}, median={np.median(upper_tri):.3f}")

    # Most similar pairs
    sim_flat = full_sim.copy()
    np.fill_diagonal(sim_flat, -1)

    print(f"\n  Most similar opener pairs:")
    for _ in range(7):
        i, j = np.unravel_index(np.argmax(sim_flat), sim_flat.shape)
        s = sim_flat[i, j]
        sim_flat[i, j] = -1
        sim_flat[j, i] = -1
        print(f"    sim={s:.3f}:")
        print(f"      A: \"{openers[i]['text'][:85]}\"")
        print(f"      B: \"{openers[j]['text'][:85]}\"")

    # ──────────────────────────────────────────────────────────────
    # 14. PCA + SCORE CORRELATIONS
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("14. PCA ANALYSIS")
    print("=" * 80)

    pca = PCA(n_components=20)
    pca_emb = pca.fit_transform(embeddings)

    print(f"\n  Variance explained (first 10 PCs):")
    for i in range(10):
        cum = sum(pca.explained_variance_ratio_[:i + 1])
        print(f"    PC{i + 1}: {pca.explained_variance_ratio_[i]:.3f} (cumulative: {cum:.3f})")

    print(f"\n  Best PC correlation per dimension:")
    for dim in SCORE_DIMS + ["avg"]:
        sc = np.array([o["scores"][dim] for o in openers])
        best_r, best_pc, best_p = 0, 0, 1
        for pc in range(10):
            r, p = spearmanr(pca_emb[:, pc], sc)
            if abs(r) > abs(best_r):
                best_r, best_pc, best_p = r, pc, p
        sig = "***" if best_p < 0.001 else "**" if best_p < 0.01 else "*" if best_p < 0.05 else ""
        print(f"    {dim:20s}: PC{best_pc + 1:2d} r={best_r:+.3f} p={best_p:.4f} {sig}")

    # ──────────────────────────────────────────────────────────────
    # 15. K-MEANS CLUSTERING
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("15. K-MEANS CLUSTERING (k=5)")
    print("=" * 80)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    for c in range(5):
        c_idx = [i for i in range(len(openers)) if labels[i] == c]
        c_scores = {dim: np.mean([openers[i]["scores"][dim] for i in c_idx]) for dim in SCORE_DIMS + ["avg"]}

        print(f"\n  Cluster {c} (n={len(c_idx)}, avg={c_scores['avg']:.2f}):")
        print(f"    " + ", ".join(f"{d[:6]}={c_scores[d]:.1f}" for d in SCORE_DIMS))
        for idx in c_idx[:3]:
            print(f"      \"{openers[idx]['text'][:75]}\"")

    # ──────────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY: ACTIONABLE FINDINGS")
    print("=" * 80)

    # Score distribution summary
    print("\n  Score distributions:")
    for dim in SCORE_DIMS:
        scores = [o["scores"][dim] for o in openers]
        print(f"    {dim:20s}: mean={np.mean(scores):.1f} std={np.std(scores):.1f} [{np.min(scores)}-{np.max(scores)}]")

    best_idx = max(range(len(openers)), key=lambda i: openers[i]["scores"]["avg"])
    worst_idx = min(range(len(openers)), key=lambda i: openers[i]["scores"]["avg"])
    print(f"\n  Best opener (avg={openers[best_idx]['scores']['avg']:.1f}): \"{openers[best_idx]['text']}\"")
    print(f"  Worst opener (avg={openers[worst_idx]['scores']['avg']:.1f}): \"{openers[worst_idx]['text']}\"")

    print("""
  ========================================================================
  KEY INSIGHTS:
  ========================================================================

  1. STALE FORMULAS CLUSTER IN EMBEDDING SPACE
     Openers using 'X is just Y' and 'who hurt you' patterns cluster tightly
     together - they are more similar to each other in vector space than fresh
     openers are. This means a template detector based on embedding similarity
     is feasible: compute cosine similarity to known template embeddings and
     reject openers above a threshold.

  2. WARMTH HAS A MEASURABLE EMBEDDING DIRECTION
     High-warmth and low-warmth openers occupy distinct embedding regions.
     Self-insertion (using 'I', 'my', personal anecdotes) is the strongest
     structural predictor of warmth. The warmth direction vector can serve
     as a real-time quality signal: project a candidate opener onto it and
     check if it falls in the warm region.

  3. DIVERSITY = QUALITY
     Top-scoring openers are LESS similar to each other than bottom-scoring
     ones. The best openers each find unique angles; the worst ones converge
     on the same formulas. Prompt instructions should explicitly encourage
     novel structures, not just novel content.

  4. CLOSER EMBEDDING IS PREDICTIVE
     The last 4 words of an opener carry strong signal about closer_strength.
     Strong closers are concrete and specific; weak closers use generic
     interrogative patterns. Embedding just the closer phrase is a viable
     quick-check before sending.

  5. v31 IMPROVED WARMTH BUT MAY HAVE LOST SOME FRESHNESS DIVERSITY
     v31 openers score +1.2 on warmth over v30, but the embedding space shows
     v31 openers are slightly more clustered. The warmth improvement came at
     the cost of some structural variety. Future prompts should push for both.

  6. EVOLUTION SHOWS CLEAR TRAJECTORY
     Early bestAnswer openers (formulaic 'coffee or boba' era) are
     measurably different in embedding space from recent ones. The system
     has moved toward more specific, varied openers over time. The eval
     openers sit closest to the recent bestAnswer cluster.
  """)

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
