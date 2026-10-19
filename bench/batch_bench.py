#!/usr/bin/env python3
"""Batched-decode sweep: N concurrent requests on mk_server.py --slots N (TP=2 library), on llama-server
--parallel N (tensor split, MTP drafts) with --engine llama, or on vLLM (docker, TP=2, MTP drafts) with --engine vllm.

For each context size and slot count a fresh server starts; each job's prompt (a different slice of
this repo's sources of ~ctx tokens plus a coding task, greedy) is first prefilled alone into its own
slot (max_tokens 1), then the N jobs are streamed at once and reuse it.  The figures cover the window
in which all N jobs decode together: total tok/s and mean per-job tok/s.  Each row also records the GPU
memory per card (after loading and peak), the KV capacity the engine reports, draft acceptance where the
engine exposes it, and the time to prefill each prompt alone; a run that does not fit is recorded as such.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 python3 bench/batch_bench.py [--slots 1,2,4,8] [--ctx 10000,60000]
      [--engine mk|llama|vllm] [--temp 1.0]   (temp > 0: top_p 0.95, top_k 20, min_p 0, the model card's thinking-mode sampling)
"""
import argparse
import json
import os
import re
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
VLLM_IMAGE = os.environ.get("VLLM_IMAGE", "vllm/vllm-openai:v0.31.0")
CHARS_PER_TOK = 2.6  # this source code tokenises at ~2.6 characters per token
TMP = os.path.join(HERE, "batch_bench_tmp")  # server logs
TASK = "Rewrite one self-contained helper function from the file above with a comment on every line."
SHORT = ["Write a Python module implementing an LRU cache with TTL expiry, thread safety and unit tests.",
         "Write a Python command-line tool that finds duplicate files by content hash, with tests.",
         "Write a Python rate limiter (token bucket) with an asyncio API and unit tests.",
         "Write a Python parser for INI files with sections, comments and type conversion, with tests.",
         "Write a Python trie with insert, delete, prefix search and unit tests.",
         "Write a Python retry decorator with exponential backoff, jitter and unit tests.",
         "Write a Python CSV to JSON converter with schema validation and unit tests.",
         "Write a Python priority task scheduler with cancellation and unit tests."]


def source_text():
    src = ""
    for f in ["qwen38_mk.cu", "batch.cuh", "prefill.cuh", "prefill_gemm.cuh", "main.cu", "capi.cpp", "README.md"]:
        src += open(os.path.join(MKDIR, f)).read() + "\n"
    return src


def prompt(src, n_tok, job, n_jobs):
    """job's own slice of the sources (no shared prefix between jobs), repeated up to ~n_tok tokens"""
    if n_tok == 0:  # peak speed: no context, one self-contained coding task per job
        return SHORT[job % len(SHORT)]
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
        elif args.engine == "vllm":
            # args.model is a Hugging Face checkpoint directory; the GPUs are the ones in CUDA_VISIBLE_DEVICES
            gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
            spec = json.dumps({"method": "mtp", "num_speculative_tokens": args.drafts})
            cmd = ["docker", "run", "--rm", "--name", "batch-bench-vllm", "--gpus", f'"device={gpus}"', "--ipc", "host",
                   "-p", f"127.0.0.1:{args.port}:8000", "-v", f"{os.path.abspath(args.model)}:/model:ro", VLLM_IMAGE,
                   "--model", "/model", "--tensor-parallel-size", "2", "--max-model-len", str(ctx),
                   "--max-num-seqs", str(slots), "--gpu-memory-utilization", "0.92", "--speculative-config", spec,
                   "--kv-cache-dtype", "fp8" if args.kv == "q8" else "auto"]
        else:
            cmd = [sys.executable, os.path.join(MKDIR, "server", "mk_server.py"), "-m", args.model, "--lib", args.lib,
                   "-c", str(ctx), "--slots", str(slots), "--kv", args.kv, "--port", str(args.port)]
        os.makedirs(TMP, exist_ok=True)
        model = os.path.basename(args.model.rstrip("/"))
        self.log = open(os.path.join(TMP, f"{args.engine}_{model}_{slots}x{ctx}_t{args.temp:g}.log"), "w")
        self.p = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(1800):
            if self.p.poll() is not None:
                self.stop()
                raise RuntimeError(f"server exited (see {self.log.name})")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except OSError:
                time.sleep(1)
        self.stop()
        raise RuntimeError("server did not start")

    def stop(self):
        if self.p.args[0] == "docker":
            subprocess.run(["docker", "stop", "batch-bench-vllm"], capture_output=True)
        self.p.terminate()
        self.p.wait(60)
        self.log.close()

    def kv_report(self):
        """the KV capacity lines the engine printed at start-up"""
        text = open(self.log.name, errors="replace").read()
        out = {}
        m = re.search(r"GPU KV cache size: ([\d,]+) tokens", text)  # vLLM
        if m:
            out["kv_tokens"] = int(m[1].replace(",", ""))
        m = re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", text)
        if m:
            out["max_concurrency"] = float(m[2])
        mib = [float(x) for x in re.findall(r"llama_kv_cache\S*: size = +([\d.]+) MiB", text)]  # llama.cpp
        if mib:
            out["kv_mib"] = sum(mib)
        lines = [ln.strip() for ln in text.splitlines() if re.search(r"\bKV\b", ln) and "mk" in ln.lower()]
        if lines:
            out["kv_line"] = lines[-1][:200]
        return out

    def spec_counters(self):
        """vLLM's cumulative (accepted, drafted) speculative tokens, None for the other engines"""
        try:
            with urllib.request.urlopen(self.url + "/metrics", timeout=5) as r:
                text = r.read().decode()
        except OSError:
            return None
        acc = sum(float(x) for x in re.findall(r"^vllm:spec_decode_num_accepted_tokens_total\{[^}]*\} (\S+)", text, re.M))
        dr = sum(float(x) for x in re.findall(r"^vllm:spec_decode_num_draft_tokens_total\{[^}]*\} (\S+)", text, re.M))
        return (acc, dr) if dr else None


class Vram:
    """samples memory.used of the visible GPUs every 0.5 s: per card after loading and peak"""

    def __init__(self):
        self.ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
        self.now, self.peak = {}, {}
        self.p = subprocess.Popen(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits",
                                   "-lms", "500"], stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self.read, daemon=True).start()

    def read(self):
        for line in self.p.stdout:
            i, mib = [x.strip() for x in line.split(",")]
            if i in self.ids:
                self.now[i] = int(mib)
                self.peak[i] = max(self.peak.get(i, 0), int(mib))

    def snapshot(self):
        time.sleep(1)
        return [self.now.get(i) for i in self.ids]

    def stop(self):
        self.p.terminate()
        return [self.peak.get(i) for i in self.ids]


def stream(url, text, n_gen, times, done, temp):
    body = {"messages": [{"role": "user", "content": text}], "max_tokens": n_gen, "temperature": temp, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}, "seed": 1, "stream_options": {"include_usage": True}}
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
                    done.append(o["timings"])  # llama-server and mk_server; usage may arrive in the same chunk
                elif o.get("usage") and not o["choices"]:  # vLLM: no timings, the final chunk carries the token count
                    done.append({"predicted_n": o["usage"]["completion_tokens"], "prompt_ms": 0.0})


def run(args, slots, ctx, src):
    row = {"engine": args.engine, "model": os.path.basename(args.model.rstrip("/")), "slots": slots, "ctx": ctx,
           "gen": args.gen}
    vram = Vram()
    try:
        srv = Server(args, slots, ctx + args.gen + 512)
    except RuntimeError as e:
        row.update(fit=False, error=str(e), vram_peak_mib=vram.stop())
        return row
    try:
        row["vram_loaded_mib"] = vram.snapshot()
        texts = [prompt(src, ctx, j, slots) for j in range(slots)]
        pre = []
        for t in texts:
            t0 = time.time()
            stream(srv.url, t, 1, [], [], args.temp)
            pre.append(time.time() - t0)
        spec0 = srv.spec_counters()
        times, done = [[] for _ in range(slots)], [[] for _ in range(slots)]
        th = [threading.Thread(target=stream, args=(srv.url, texts[j], args.gen, times[j], done[j], args.temp))
              for j in range(slots)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        spec1 = srv.spec_counters()
    finally:
        srv.stop()
        row["vram_peak_mib"] = vram.stop()
        row.update(srv.kv_report())
    row["prefill_alone_s"] = [round(x, 2) for x in pre]
    row["gen_tokens"] = [d[0]["predicted_n"] for d in done]
    if spec0 and spec1:
        row["accept"] = (spec1[0] - spec0[0]) / max(spec1[1] - spec0[1], 1)
    elif all("draft_n" in d[0] for d in done):
        row["accept"] = sum(d[0]["draft_n_accepted"] for d in done) / max(sum(d[0]["draft_n"] for d in done), 1)
    # deltas are text pieces, not tokens: scale each job's count by its token total from the timings
    w0, w1 = max(t[0] for t in times), min(t[-1] for t in times)
    if w1 <= w0:
        row.update(fit=False, error="the jobs never decoded together (queued: KV cache full, or too short)")
        return row
    tok = [d[0]["predicted_n"] * sum(w0 <= x <= w1 for x in t) / len(t) for t, d in zip(times, done)]
    span = w1 - w0
    row.update(fit=True, total_tok_s=sum(tok) / span, job_tok_s=sum(tok) / span / slots, window_s=span)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--lib", default=LIB)
    ap.add_argument("--slots", default="1,2,4,8")
    ap.add_argument("--ctx", default="10000,60000")
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--kv", default="q8", choices=["auto", "f16", "q8"])
    ap.add_argument("--port", type=int, default=18099)
    ap.add_argument("--engine", default="mk", choices=["mk", "llama", "vllm"])
    ap.add_argument("--drafts", type=int, default=4, help="vLLM MTP draft tokens (Geezuz runs -d 4)")
    ap.add_argument("--temp", type=float, default=0.0)
    args = ap.parse_args()
    src = source_text()
    rows = []
    for ctx in map(int, args.ctx.split(",")):
        for slots in map(int, args.slots.split(",")):
            try:
                r = run(args, slots, ctx, src)
            except Exception as e:  # one failed run must not end the sweep
                r = {"engine": args.engine, "model": os.path.basename(args.model.rstrip("/")), "slots": slots,
                     "ctx": ctx, "fit": False, "error": repr(e)[:300], "vram_peak_mib": None}
            rows.append(r)
            print(json.dumps(r), flush=True)
    print("\n| ctx | slots | total tok/s | per job tok/s | VRAM peak MiB |\n|---|---|---|---|---|")
    for r in rows:
        tps = f"{r['total_tok_s']:.1f} | {r['job_tok_s']:.1f}" if r["fit"] else "does not fit | "
        print(f"| {r['ctx']} | {r['slots']} | {tps} | {r['vram_peak_mib']} |")


if __name__ == "__main__":
    main()
