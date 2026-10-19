#!/usr/bin/env python3
"""Two-GPU benchmark: the TP=2 megakernel vs llama-server on the same two cards.

  1. decode on the 4 chat prompts of prompts.txt (greedy, 512 tokens), with each engine's
     draft acceptance (llama.cpp: accepted / drafted; megakernel: tokens per cycle)
  2. context sweep: a source file of ~N tokens pasted into the prompt plus a coding task;
     prompt-processing and decode speed at each N

llama-server runs in each --llama-configs entry, "<split mode>:<MTP drafts>" (default: layer:2,
open-jet's flags; layer:4; tensor:4). It must be a CUDA build (LLAMA_SERVER, default
../llama.cpp/build-cuda/bin/llama-server). Both engines use the GPUs CUDA makes visible:

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 python3 bench/tp2_bench.py
"""
import argparse
import json
import os
import re
import subprocess
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MKDIR = os.path.dirname(HERE)
ROOT = os.path.dirname(MKDIR)
MODEL = os.path.join(ROOT, "models", "Qwen3.8-27B-Q4_K_M.gguf")
LLAMA_SERVER = os.environ.get("LLAMA_SERVER", os.path.join(ROOT, "llama.cpp", "build-cuda", "bin", "llama-server"))
MK = os.path.join(MKDIR, "geezuz_tp2")
DV = os.path.join(MKDIR, "draft_vocab_32k.bin")
TMP = os.path.join(HERE, "tp2_bench_tmp")
CHARS_PER_TOK = 2.6  # this source code tokenises at ~2.6 characters per token

TASK = "Write a C++ unit test (plain asserts, no framework) for one self-contained helper function from the file above."


def chat(p):
    return f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def code_text(n_tok):
    src = ""
    while len(src) < n_tok * CHARS_PER_TOK:
        for f in ["qwen38_mk.cu", "prefill.cuh", "prefill_gemm.cuh", "main.cu", "README.md"]:
            src += open(os.path.join(MKDIR, f)).read() + "\n"
    return src[: int(n_tok * CHARS_PER_TOK)]


def ctx_prompt(n_tok):
    if n_tok == 0:
        return "Write a C++ function that parses a GGUF file header (magic, version, tensor count) from a byte buffer, with error checks."
    return "Here is a source file:\n\n" + code_text(n_tok) + "\n\n" + TASK


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


class Llama:
    def __init__(self, split, n_draft, ctx, port=8093):
        self.url = f"http://127.0.0.1:{port}"
        cmd = [LLAMA_SERVER, "-m", MODEL, "--host", "127.0.0.1", "--port", str(port), "--parallel", "1",
               "--flash-attn", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-ngl", "99", "-c", str(ctx),
               "--spec-type", "draft-mtp", "--spec-draft-n-max", str(n_draft), "-sm", split]
        self.log = open(os.path.join(TMP, f"llama_server_{split}_{n_draft}.log"), "w")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(900):
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited, see {self.log.name}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(1)
        raise RuntimeError("llama-server did not start")

    def complete(self, prompt, n):
        r = post(self.url + "/completion", {"prompt": prompt, "n_predict": n, "temperature": 0.0, "top_k": 1,
                                            "cache_prompt": False})
        return r["timings"]

    def close(self):
        self.proc.terminate()
        self.proc.wait()


def mk_run(args, ctx):
    out = subprocess.run([MK, "-m", MODEL, "-c", str(ctx), "--draft-vocab", DV, "--quiet"] + args,
                         capture_output=True, text=True)
    t = re.findall(r"prompt (\d+) tok in ([\d.]+) s \(([\d.]+) tok/s\)\n.*decode (\d+) tok in ([\d.]+) s = ([\d.]+) tok/s"
                   r".*?, ([\d.]+) tok/cycle", out.stderr)
    if not t:
        raise RuntimeError(out.stderr[-2000:])
    t = t[0]
    return dict(pn=int(t[0]), ptps=float(t[2]), dtps=float(t[5]), tpc=float(t[6]))


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx-sizes", default="0,4000,16000,30000,60000")
    ap.add_argument("--llama-configs", default="layer:2,layer:4,tensor:4")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("-n", type=int, default=256, help="tokens generated per context-sweep run")
    ap.add_argument("-c", "--ctx", type=int, default=64000, help="KV capacity of both engines")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    MODEL = args.model
    os.makedirs(TMP, exist_ok=True)
    sizes = [int(x) for x in args.ctx_sizes.split(",")]
    prompts = [l.strip() for l in open(os.path.join(HERE, "prompts.txt")) if l.strip()]
    files = {}
    for n in sizes:
        files[n] = os.path.join(TMP, f"ctx_{n}.txt")
        open(files[n], "w").write(ctx_prompt(n))

    res = {}
    engines = []
    for cfg in args.llama_configs.split(","):
        split, n_draft = cfg.split(":")
        lab = f"llama.cpp {split} MTP {n_draft}"
        try:
            ll = Llama(split, int(n_draft), args.ctx)
        except RuntimeError as e:
            print(f"[{lab}] skipped: {e}", flush=True)
            continue
        try:
            for i, p in enumerate(prompts):
                t = max((ll.complete(chat(p), 512) for _ in range(args.reps)), key=lambda x: x["predicted_per_second"])
                res[("short", lab, i)] = t["predicted_per_second"]
                acc = t.get("draft_n_accepted", 0) / max(t.get("draft_n", 0), 1)
                print(f"[{lab}] prompt{i}: {t['predicted_per_second']:.1f} tok/s, {acc:.1%} of drafts accepted",
                      flush=True)
            for n in sizes:
                runs = [ll.complete(chat(open(files[n]).read()), args.n) for _ in range(args.reps)]
                r = dict(pn=runs[0]["prompt_n"], ptps=max(x["prompt_per_second"] for x in runs),
                         dtps=max(x["predicted_per_second"] for x in runs))
                res[("ctx", lab, n)] = r
                print(f"[{lab}] ctx {r['pn']} tok: prompt {r['ptps']:.0f} tok/s, decode {r['dtps']:.1f} tok/s", flush=True)
            engines.append(lab)
        except Exception as e:
            print(f"[{lab}] failed: {e}", flush=True)
        finally:
            ll.close()

    lab = "megakernel TP2 -d 4"
    engines.append(lab)
    for i, p in enumerate(prompts):
        t = max((mk_run(["-p", p, "-n", "512", "-d", "4"], args.ctx) for _ in range(args.reps)), key=lambda x: x["dtps"])
        res[("short", lab, i)] = t["dtps"]
        print(f"[mk] prompt{i}: {t['dtps']:.1f} tok/s, {t['tpc']:.2f} tok/cycle", flush=True)
    for n in sizes:
        runs = [mk_run(["-f", files[n], "-n", str(args.n), "-d", "4"], args.ctx) for _ in range(args.reps)]
        r = dict(pn=runs[0]["pn"], ptps=max(x["ptps"] for x in runs), dtps=max(x["dtps"] for x in runs))
        res[("ctx", lab, n)] = r
        print(f"[mk] ctx {r['pn']} tok: prompt {r['ptps']:.0f} tok/s, decode {r['dtps']:.1f} tok/s", flush=True)

    # ---------------- tables ----------------
    print("\n## Decode, short chat prompts (tok/s, best of %d)\n" % args.reps)
    print("| prompt | " + " | ".join(engines) + " |\n|" + "---|" * (len(engines) + 1))
    for i, p in enumerate(prompts):
        print(f"| {p[:40]}... | " + " | ".join(f"{res[('short', e, i)]:.1f}" for e in engines) + " |")
    print("| **mean** | " + " | ".join(f"**{sum(res[('short', e, i)] for i in range(len(prompts))) / len(prompts):.1f}**"
                                        for e in engines) + " |")
    for what, key in (("Prompt processing", "ptps"), ("Decode", "dtps")):
        print(f"\n## Code in context: {what.lower()} (tok/s, {args.n} generated tokens)\n")
        print("| context (tok) | " + " | ".join(engines) + " |\n|" + "---|" * (len(engines) + 1))
        for n in sizes:
            rr = [res[("ctx", e, n)] for e in engines]
            print(f"| {rr[-1]['pn']} | " + " | ".join(f"{r[key]:.0f}" if key == "ptps" else f"{r[key]:.1f}"
                                                   for r in rr) + " |")


if __name__ == "__main__":
    main()
