#!/usr/bin/env python3
"""Output-quality check through each engine's OpenAI server: GSM8K accuracy (the first --n questions of the
test set, greedy, thinking off) and a few sample answers saved in full for side-by-side reading.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 python3 bench/quality_bench.py --engine mk|llama|vllm
      [--model ...] [--n 200] [--slots 8] [--gsm8k test.jsonl] [--out results_dir]

The test set is openai/grade-school-math grade_school_math/data/test.jsonl.  An answer counts when the last
number in the reply equals the reference after "####".
"""
import argparse
import json
import os
import re
import threading
import urllib.request

from batch_bench import LIB, MODEL, Server

PROMPT = "Solve this math problem step by step. End with a line of the form 'Answer: <number>'.\n\n{q}"
SAMPLES = [
    "Write a Python function that merges overlapping intervals, with a short docstring.",
    "Explain in three sentences why the sky is blue.",
    "Write a four-line poem about a lighthouse.",
]


def chat(url, text, n_gen):
    body = {"messages": [{"role": "user", "content": text}], "max_tokens": n_gen, "temperature": 0.0, "seed": 1,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def number(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s)
    return m[-1].replace(",", "").rstrip(".") if m else None


def same(a, b):
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="mk", choices=["mk", "llama", "vllm"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--lib", default=LIB)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--gen", type=int, default=1024)
    ap.add_argument("--kv", default="q8", choices=["auto", "f16", "q8"])
    ap.add_argument("--port", type=int, default=18099)
    ap.add_argument("--drafts", type=int, default=4)
    ap.add_argument("--gsm8k", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsm8k_test.jsonl"))
    ap.add_argument("--out", default="quality_out")
    args = ap.parse_args()
    args.temp = 0.0
    qs = [json.loads(line) for line in open(args.gsm8k)][: args.n]
    tag = f"{args.engine}_{os.path.basename(args.model.rstrip('/'))}"
    os.makedirs(args.out, exist_ok=True)
    srv = Server(args, args.slots, 4096)
    try:
        replies, lock, nxt = [None] * len(qs), threading.Lock(), [0]

        def worker():
            while True:
                with lock:
                    i = nxt[0]
                    nxt[0] += 1
                if i >= len(qs):
                    return
                replies[i] = chat(srv.url, PROMPT.format(q=qs[i]["question"]), args.gen)

        th = [threading.Thread(target=worker) for _ in range(args.slots)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        samples = [chat(srv.url, p, 512) for p in SAMPLES]
    finally:
        srv.stop()
    rows = []
    for q, r in zip(qs, replies):
        ref = q["answer"].split("####")[-1].strip().replace(",", "")
        rows.append({"ref": ref, "got": number(r), "ok": same(number(r), ref), "reply": r})
    ok = sum(r["ok"] for r in rows)
    with open(os.path.join(args.out, f"gsm8k_{tag}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out, f"samples_{tag}.md"), "w") as f:
        for p, s in zip(SAMPLES, samples):
            f.write(f"## {p}\n\n{s}\n\n")
    print(f"{tag}: GSM8K {ok}/{len(rows)} ({100 * ok / len(rows):.1f}%)")


if __name__ == "__main__":
    main()
