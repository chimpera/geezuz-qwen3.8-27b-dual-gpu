#!/usr/bin/env python3
"""Batched-decode sweep: N concurrent requests on mk_server.py --slots N (TP=2 library), or on llama-server
--parallel N (tensor split, MTP drafts) with --engine llama.

For each context size and slot count a fresh server starts; each job's prompt (a different slice of
this repo's sources of ~ctx tokens plus a coding task, greedy) is first prefilled alone into its own
slot (max_tokens 1), then the N jobs are streamed at once and reuse it.  The figures cover the window
in which all N jobs decode together: total tok/s and mean per-job tok/s.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 python3 bench/batch_bench.py [--slots 1,2,3,4,6,8] [--ctx 10000,60000]
      [--engine mk|llama] [--temp 1.0]   (temp > 0: top_p 0.95, top_k 20, min_p 0, the model card's thinking-mode sampling)
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MKDIR = os.path.dirname(HERE)
ROOT = os.path.dirname(MKDIR)
MODEL = os.path.join(ROOT, "models", "Qwen3.8-27B-Q8_0.gguf")
LIB = os.path.join(MKDIR, "libgeezuz_tp2.so")
LLAMA_SERVER = os.environ.get("LLAMA_SERVER", os.path.join(ROOT, "llama.cpp", "build-cuda", "bin", "llama-server"))
CHARS_PER_TOK = 2.6  # this source code tokenises at ~2.6 characters per token
KV_POOL = 400_000  # q8_0 KV positions that fit next to the Q8_0 weights per card (all slots)
TMP = os.path.join(HERE, "batch_bench_tmp")  # server logs
TASK = "Rewrite one self-contained helper function from the file above with a comment on every line."


def source_text():
    src = ""
    for f in ["qwen38_mk.cu", "batch.cuh", "prefill.cuh", "prefill_gemm.cuh", "main.cu", "capi.cpp", "README.md"]:
        src += open(os.path.join(MKDIR, f)).read() + "\n"
    return src


def prompt(src, n_tok, job, n_jobs):
    """job's own slice of the sources (no shared prefix between jobs), repeated up to ~n_tok tokens"""
    n = int(n_tok * CHARS_PER_TOK)
    start = job * len(src) // n_jobs
    text = ""
    while len(text) < n:
        text += src[start:] + src[:start]
    return "Here is a source file:\n\n" + text[:n] + "\n\n" + TASK


class Server:
    def __init__(self, args, slots, ctx):
        self.url = f"http://127.0.0.1:{args.port}"
        if args.engine == "llama":
            # llama-server splits its context over the slots
            cmd = [LLAMA_SERVER, "-m", args.model, "--host", "127.0.0.1", "--port", str(args.port), "--parallel", str(slots),
                   "-c", str(slots * ctx), "--flash-attn", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-ngl", "99",
                   "--spec-type", "draft-mtp", "--spec-draft-n-max", "4", "-sm", "tensor"]
        else:
            cmd = [sys.executable, os.path.join(MKDIR, "server", "mk_server.py"), "-m", args.model, "--lib", args.lib,
                   "-c", str(ctx), "--slots", str(slots), "--kv", args.kv, "--port", str(args.port)]
        os.makedirs(TMP, exist_ok=True)
        self.log = open(os.path.join(TMP, f"{args.engine}_{slots}x{ctx}_t{args.temp:g}.log"), "w")
        self.p = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(600):
            if self.p.poll() is not None:
                raise RuntimeError(f"server exited (see {self.log.name})")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except OSError:
                time.sleep(1)
        raise RuntimeError("server did not start")

    def stop(self):
        self.p.terminate()
        self.p.wait(60)
        self.log.close()


def stream(url, text, n_gen, times, done, temp):
    body = {"messages": [{"role": "user", "content": text}], "max_tokens": n_gen, "temperature": temp, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}, "seed": 1}
    if temp > 0:
        body.update({"top_p": 0.95, "top_k": 20, "min_p": 0.0})
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            if line.startswith(b"data: {"):
                o = json.loads(line[6:])
                if o["choices"] and o["choices"][0]["delta"].get("content"):
                    times.append(time.time())
                if o.get("timings"):
                    done.append(o["timings"])


def run(args, slots, ctx, src):
    srv = Server(args, slots, ctx + args.gen + 512)
    try:
        texts = [prompt(src, ctx, j, slots) for j in range(slots)]
        for t in texts:
            stream(srv.url, t, 1, [], [], args.temp)
        times, done = [[] for _ in range(slots)], [[] for _ in range(slots)]
        th = [threading.Thread(target=stream, args=(srv.url, texts[j], args.gen, times[j], done[j], args.temp))
              for j in range(slots)]
        for t in th:
            t.start()
        for t in th:
            t.join()
    finally:
        srv.stop()
    # deltas are text pieces, not tokens: scale each job's count by its token total from the timings
    w0, w1 = max(t[0] for t in times), min(t[-1] for t in times)
    if w1 <= w0:
        return None
    tok = [d[0]["predicted_n"] * sum(w0 <= x <= w1 for x in t) / len(t) for t, d in zip(times, done)]
    span = w1 - w0
    return {"slots": slots, "ctx": ctx, "total_tok_s": sum(tok) / span, "job_tok_s": sum(tok) / span / slots,
            "window_s": span, "reprefill_s": max(d[0]["prompt_ms"] for d in done) / 1e3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--lib", default=LIB)
    ap.add_argument("--slots", default="1,2,3,4,6,8")
    ap.add_argument("--ctx", default="10000,60000")
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--kv", default="q8", choices=["auto", "f16", "q8"])
    ap.add_argument("--port", type=int, default=18099)
    ap.add_argument("--engine", default="mk", choices=["mk", "llama"])
    ap.add_argument("--temp", type=float, default=0.0)
    args = ap.parse_args()
    src = source_text()
    rows = []
    for ctx in map(int, args.ctx.split(",")):
        for slots in map(int, args.slots.split(",")):
            if slots * (ctx + args.gen + 512) > KV_POOL:
                print(f"skip {slots} x {ctx}: KV pool", flush=True)
                continue
            r = run(args, slots, ctx, src)
            if r is None:
                print(f"{slots} x {ctx}: jobs never decoded together (raise --gen)", flush=True)
                continue
            rows.append(r)
            print(json.dumps(r), flush=True)
    print("\n| ctx | slots | total tok/s | per job tok/s |\n|---|---|---|---|")
    for r in rows:
        print(f"| {r['ctx']} | {r['slots']} | {r['total_tok_s']:.1f} | {r['job_tok_s']:.1f} |")


if __name__ == "__main__":
    main()
