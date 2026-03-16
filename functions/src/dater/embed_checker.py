#!/usr/bin/env python3
"""
Embedding-based opener quality checker.

Runs as a lightweight HTTP server on port 5201 so the model stays loaded
across calls (cold model load ~2-3s, warm request ~20ms).

POST /check
  Body: JSON with openers, templates, generic_closers, accepted
  Returns: JSON with similarity scores

GET /healthcheck
  Returns: {"status": "ok"}

Can also be invoked as a one-shot CLI tool via stdin/stdout (slower due
to model reload each time):
  echo '{"openers": [...], ...}' | python3 embed_checker.py --stdin
"""

import json
import sys
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
PORT = 5201

# Load model once at module level
_model = SentenceTransformer(MODEL_NAME)


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384))
    return _model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


def process_request(data: dict) -> dict:
    openers = data["openers"]
    templates = data.get("templates", [])
    generic_closers = data.get("generic_closers", [])
    accepted = data.get("accepted", [])

    if not openers:
        return {"results": []}

    # Embed everything in batch for efficiency
    opener_embs = embed(openers)
    template_embs = embed(templates) if templates else np.zeros((0, 384))
    accepted_embs = embed(accepted) if accepted else np.zeros((0, 384))

    # Embed closers (last 4 words of each opener)
    closer_texts = []
    for o in openers:
        words = o.split()
        closer_texts.append(" ".join(words[-4:]) if len(words) >= 4 else o)
    closer_embs = embed(closer_texts)

    generic_closer_embs = embed(generic_closers) if generic_closers else np.zeros((0, 384))

    results = []
    for i, opener in enumerate(openers):
        # 1. Max template similarity
        template_sim = 0.0
        if len(template_embs) > 0:
            sims = opener_embs[i] @ template_embs.T
            template_sim = float(np.max(sims))

        # 2. Closer specificity = 1 - max similarity to generic closers
        #    High value = more specific (good), low value = too generic (bad)
        closer_specificity = 1.0
        if len(generic_closer_embs) > 0:
            closer_sims = closer_embs[i] @ generic_closer_embs.T
            closer_specificity = 1.0 - float(np.max(closer_sims))

        # 3. Pairwise max similarity to already-accepted openers
        pairwise_max = 0.0
        if len(accepted_embs) > 0:
            pair_sims = opener_embs[i] @ accepted_embs.T
            pairwise_max = float(np.max(pair_sims))

        results.append({
            "opener": opener,
            "template_similarity": round(template_sim, 4),
            "closer_specificity": round(closer_specificity, 4),
            "pairwise_max_similarity": round(pairwise_max, 4),
        })

    return {"results": results}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthcheck":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/check":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                result = process_request(data)
                response = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)
            except Exception as e:
                err = json.dumps({"error": str(e)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(err)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default request logging to keep output clean
        pass


def run_server():
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"embed_checker server listening on http://127.0.0.1:{PORT}")
    sys.stdout.flush()
    server.serve_forever()


def run_stdin():
    data = json.load(sys.stdin)
    result = process_request(data)
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    if "--stdin" in sys.argv:
        run_stdin()
    else:
        run_server()
