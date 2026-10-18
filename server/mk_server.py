#!/usr/bin/env python3
"""OpenAI-compatible chat server for the Qwen3.8-27B megakernel (a drop-in for the parts of
llama-server's API that open-jet uses).

  python3 server/mk_server.py -m model.gguf --port 18080 [--ctx 32768] [--draft 4]

POST /v1/chat/completions (streaming or not): the request is rendered with the chat template stored
in the GGUF (enable_thinking / preserve_thinking / reasoning_effort via chat_template_kwargs, tools
via `tools`), reasoning is returned as `reasoning_content`, Qwen's XML tool calls as structured
`tool_calls`.  Sampling: temperature, top_p, top_k (<= 32), min_p, seed.  Successive requests of a
conversation reuse the processed tokens: the new prompt is matched against the sequence so far and
only the rest is prefilled (rewinding to the end of the previous prompt if it diverges after that).
GET /health, GET /v1/models.  --slots N decodes up to N requests at once (each in its own sequence slot,
picked by the longest cached prefix); more wait for a free slot.  A prompt is prefilled between decode cycles, in chunks while others decode.
GET /status: what the engine is doing now (phase, live token count, tail of the reasoning / text / tool call
being generated, per slot under `slots`) and the stats of the last finished request.
"""
import argparse
import codecs
import ctypes
import json
import os
import queue
import random
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

HERE = os.path.dirname(os.path.abspath(__file__))
# Forced into the reasoning when the thinking budget runs out, so the model answers instead of hitting max_tokens.
THINK_STOP = "\n\nI have thought enough and will answer now based on the reasoning so far.\n</think>\n\n"
TOPK_MAX = 32
REUSE_MIN = 128  # cached prefix (tokens) worth steering a request to a slot; shorter ones re-prefill in a blink
STATUS_TAIL = 400  # characters of reasoning / text / tool call (or of a tool argument) GET /status shows


def log(*a):
    print("[mk-server]", *a, file=sys.stderr, flush=True)


class Engine:
    """ctypes binding of capi.cpp"""

    def __init__(self, lib, model, ctx, n_draft, draft_vocab, kv_mode=-1, n_seq=1):
        L = self.L = ctypes.CDLL(lib)
        vp, ip, cp = ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p
        sig = {
            "mk_new": (vp, [cp, ctypes.c_int, ctypes.c_int, cp, ctypes.c_int, ctypes.c_int]),
            "mk_last_error": (cp, []),
            "mk_chat_template": (ctypes.c_int, [vp, cp, ctypes.c_int]),
            "mk_tokenize": (ctypes.c_int, [vp, cp, ctypes.c_int, ip, ctypes.c_int]),
            "mk_piece": (ctypes.c_int, [vp, ctypes.c_int, cp, ctypes.c_int]),
            "mk_is_eog": (ctypes.c_int, [vp, ctypes.c_int]),
            "mk_set_sampling": (ctypes.c_int, [vp, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float,
                                               ctypes.c_ulonglong]),
            "mk_prefill": (ctypes.c_int, [vp, ip, ctypes.c_int]),
            "mk_append": (ctypes.c_int, [vp, ip, ctypes.c_int]),
            "mk_step": (ctypes.c_int, [vp, ip, ctypes.c_int]),
            "mk_select": (ctypes.c_int, [vp, ctypes.c_int]),
            "mk_step_batch": (ctypes.c_int, [vp, ip, ctypes.c_int, ip, ctypes.c_int, ip]),
            "mk_reset": (None, [vp]),
            "mk_truncate": (ctypes.c_int, [vp, ctypes.c_int]),
            "mk_sequence": (ctypes.c_int, [vp, ip, ctypes.c_int]),
            "mk_committed": (ctypes.c_int, [vp]),
            "mk_checkpoint_pos": (ctypes.c_int, [vp]),
            "mk_rewind_to_checkpoint": (ctypes.c_int, [vp]),
        }
        for name, (res, args) in sig.items():
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        self.h = L.mk_new(model.encode(), ctx, n_draft, draft_vocab.encode() if draft_vocab else None, kv_mode,
                          n_seq)
        if not self.h:
            raise RuntimeError(L.mk_last_error().decode())
        self.ctx = ctx
        self._pieces = {}

    def _err(self, what):
        return RuntimeError(f"{what}: {self.L.mk_last_error().decode()}")

    def chat_template(self):
        n = self.L.mk_chat_template(self.h, None, 0)
        buf = ctypes.create_string_buffer(n + 1)
        self.L.mk_chat_template(self.h, buf, n)
        return buf.raw[:n].decode()

    def tokenize(self, text):
        b = text.encode()
        cap = len(b) + 16
        out = (ctypes.c_int * cap)()
        n = self.L.mk_tokenize(self.h, b, len(b), out, cap)
        if n < 0:
            cap = -n
            out = (ctypes.c_int * cap)()
            n = self.L.mk_tokenize(self.h, b, len(b), out, cap)
        return list(out[:n])

    def piece(self, tok):
        p = self._pieces.get(tok)
        if p is None:
            buf = ctypes.create_string_buffer(256)
            n = self.L.mk_piece(self.h, tok, buf, 256)
            p = self._pieces[tok] = buf.raw[:max(n, 0)]
        return p

    def is_eog(self, tok):
        return bool(self.L.mk_is_eog(self.h, tok))

    def set_sampling(self, temp, top_p, top_k, min_p, seed):
        if self.L.mk_set_sampling(self.h, temp, top_p, top_k, min_p, seed) < 0:
            raise self._err("sampling")

    def prefill(self, toks):
        arr = (ctypes.c_int * len(toks))(*toks)
        r = self.L.mk_prefill(self.h, arr, len(toks))
        if r < 0:
            raise self._err("prefill")
        return r

    def append(self, toks):
        arr = (ctypes.c_int * len(toks))(*toks)
        r = self.L.mk_append(self.h, arr, len(toks))
        if r < 0:
            raise self._err("append")
        return r

    def step(self):
        out = (ctypes.c_int * 8)()
        n = self.L.mk_step(self.h, out, 8)
        if n < 0:
            raise self._err("decode")
        return list(out[:n])

    def select(self, s):
        if self.L.mk_select(self.h, s) < 0:
            raise self._err("select")

    def step_batch(self, slots):
        """one cycle of the given slots; returns each slot's tokens"""
        n, cap = len(slots), 8
        out, counts = (ctypes.c_int * (n * cap))(), (ctypes.c_int * n)()
        if self.L.mk_step_batch(self.h, (ctypes.c_int * n)(*slots), n, out, cap, counts) < 0:
            raise self._err("decode")
        return [list(out[b * cap:b * cap + counts[b]]) for b in range(n)]

    def reset(self):
        self.L.mk_reset(self.h)

    def truncate(self, n):
        if self.L.mk_truncate(self.h, n) < 0:
            raise self._err("truncate")

    def sequence(self):
        n = self.L.mk_sequence(self.h, None, 0)
        out = (ctypes.c_int * max(n, 1))()
        self.L.mk_sequence(self.h, out, n)
        return list(out[:n])

    def committed(self):
        return self.L.mk_committed(self.h)

    def checkpoint_pos(self):
        return self.L.mk_checkpoint_pos(self.h)

    def rewind_to_checkpoint(self):
        return self.L.mk_rewind_to_checkpoint(self.h) == 0


# ---------------------------------------------------------------------------------------------
# chat template and output parsing


def make_template(src):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, sort_keys=False: json.dumps(
        x, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys)
    return env.from_string(src)


class BadRequest(Exception):
    pass


def normalize_messages(messages):
    out = []
    for m in messages:
        m = dict(m)
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") not in (None, "text"):
                    raise BadRequest(f"unsupported content part type {part.get('type')!r} (text only)")
        if content is None:
            m["content"] = ""
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {}
                fn["arguments"] = args if isinstance(args, dict) else {}
                tc["function"] = fn
                calls.append(tc)
            m["tool_calls"] = calls
        out.append(m)
    return out


TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)(?:</function>\s*(?:</tool_call>|$)|$)", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_tool_calls(text, tools):
    schemas = {}
    for t in tools or []:
        fn = t.get("function", t)
        schemas[fn.get("name")] = (fn.get("parameters") or {}).get("properties") or {}
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        name = m.group(1).strip()
        props = schemas.get(name, {})
        args = {}
        for p in PARAM_RE.finditer(m.group(2)):
            key, val = p.group(1).strip(), p.group(2)
            typ = (props.get(key) or {}).get("type")
            if typ == "string" or (typ is None and not val.strip()[:1] in "[{0123456789-tfn"):
                args[key] = val
            else:
                try:
                    args[key] = json.loads(val)
                except json.JSONDecodeError:
                    args[key] = val
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return calls


class OutputParser:
    """Splits the generated text into reasoning (before </think>), content and tool-call markup."""

    def __init__(self, thinking, tools):
        self.mode = "reasoning" if thinking else "content_start"
        self.tools = tools
        self.buf = ""
        self.tool_text = ""

    def feed(self, text, final=False):
        """returns [(kind, text)] with kind in {reasoning, content}"""
        out = []
        self.buf += text
        while True:
            if self.mode == "reasoning":
                i = self.buf.find("</think>")
                if i >= 0:
                    out.append(("reasoning", self.buf[:i]))
                    self.buf = self.buf[i + len("</think>"):]
                    self.mode = "content_start"
                    continue
                keep = 0 if final else len("</think>") - 1
                if len(self.buf) > keep:
                    out.append(("reasoning", self.buf[:len(self.buf) - keep]))
                    self.buf = self.buf[len(self.buf) - keep:]
                break
            if self.mode == "content_start":
                self.buf = self.buf.lstrip()
                if not self.buf:
                    break
                self.mode = "content"
                continue
            if self.mode == "content":
                if self.tools:
                    i = self.buf.find("<tool_call>")
                    if i >= 0:
                        out.append(("content", self.buf[:i].rstrip()))
                        self.tool_text = self.buf[i:]
                        self.buf = ""
                        self.mode = "tools"
                        break
                    keep = 0 if final else len("<tool_call>") - 1
                else:
                    keep = 0
                if final:
                    out.append(("content", self.buf.rstrip()))
                    self.buf = ""
                elif len(self.buf) > keep:
                    # trailing whitespace waits for more text: it is dropped before a tool call or the end
                    seg = self.buf[:len(self.buf) - keep]
                    head = seg.rstrip()
                    out.append(("content", head))
                    self.buf = seg[len(head):] + self.buf[len(self.buf) - keep:]
                break
            # tools: collect everything
            self.tool_text += self.buf
            self.buf = ""
            break
        return [(k, t) for k, t in out if t]


# ---------------------------------------------------------------------------------------------


class Job:
    """one chat completion; the scheduler thread runs it, the handler thread reads its events"""

    def __init__(self, body, toks, thinking, max_new, budget):
        self.body, self.toks, self.max_new, self.budget = body, toks, max_new, budget
        self.tools = body.get("tools") or []
        self.parser = OutputParser(thinking, self.tools)
        self.dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.tail, self.shown = {"reasoning": "", "content": ""}, 0.0
        self.n_gen, self.n_think, self.finish, self.reused = 0, 0, "length", 0
        self.events = queue.Queue()
        self.cancelled = False  # set by the handler when the client is gone
        self.slot = None
        self.fed = 0  # prompt tokens the slot holds
        self.t0 = self.t1 = time.time()

    def show(self, kind, delta):
        self.tail[kind] = (self.tail[kind] + delta)[-STATUS_TAIL:]
        self.events.put(("delta", kind, delta))

    def wait(self, emit):
        """handler side: passes the deltas to emit until the job ends; returns (finish, calls, usage, timings)"""
        while True:
            ev = self.events.get()
            if ev[0] == "delta":
                if not self.cancelled and emit(ev[1], ev[2]) is False:
                    self.cancelled = True
            elif ev[0] == "done":
                return ev[1]
            else:
                raise ev[1]


class Server:
    def __init__(self, args):
        self.args = args
        t0 = time.time()
        self.eng = Engine(args.lib, args.model, args.ctx, args.draft, args.draft_vocab,
                          {"f16": 0, "q8": 1, "auto": -1}[args.kv], args.slots)
        self.template = make_template(self.eng.chat_template())
        self.warm_up()
        self.model_id = os.path.basename(args.model)
        self.warned = set()
        self.queue = queue.Queue()
        self.active = {}  # slot -> Job, touched by the scheduler thread only
        self.slot_seq = [[] for _ in range(args.slots)]  # each slot's sequence when its last job ended
        self.slot_used = [0.0] * args.slots
        now = time.time()
        self.slot_status = [{"slot": s, "busy": False, "phase": "idle", "action": None, "updated": now}
                            for s in range(args.slots)]
        self.last = None
        threading.Thread(target=self.run, name="scheduler", daemon=True).start()
        log(f"ready in {time.time() - t0:.1f} s: {self.model_id}, ctx {args.ctx}, {args.draft} drafts, "
            f"{args.slots} slot(s)")

    def warm_up(self):
        """Runs one prefill and one decode cycle before the server listens.

        Loading allocates the weights and the KV cache, but the first kernel launches still need
        device memory of their own. When the context is too large for that, loading succeeds, /health
        answers, and the first request ends the process with a CUDA out-of-memory (CK exits), which
        the client sees as a dropped connection. Warming up here moves that failure to startup.
        The prompt must be at least two tokens: only then does prefill take the batched path, whose
        buffers are allocated on first use; with several slots the cycle is a batched one."""
        toks = self.eng.tokenize("Hello, how are you today?")
        assert len(toks) >= 2
        slots = list(range(min(self.args.slots, 2)))
        for s in slots:
            self.eng.select(s)
            self.eng.prefill(toks)
        self.eng.step_batch(slots)
        for s in slots:
            self.eng.select(s)
            self.eng.reset()

    def warn_once(self, key, msg):
        if key not in self.warned:
            self.warned.add(key)
            log("warning:", msg)

    def render(self, body):
        kw = dict(body.get("chat_template_kwargs") or {})
        ctx = dict(messages=normalize_messages(body.get("messages") or []), add_generation_prompt=True)
        if body.get("tools"):
            ctx["tools"] = body["tools"]
        for k in ("enable_thinking", "preserve_thinking", "reasoning_effort"):
            if k in kw:
                ctx[k] = kw[k]
        try:
            text = self.template.render(**ctx)
        except jinja2.exceptions.TemplateError as e:
            raise BadRequest(f"chat template: {e}")
        thinking = text.endswith("<think>\n")
        return text, thinking

    def prepare(self, body):
        """handler side: renders and tokenizes the request (BadRequest before anything is streamed)"""
        text, thinking = self.render(body)
        toks = self.eng.tokenize(text)
        max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or body.get("n_predict") or -1)
        limit = self.args.ctx - len(toks) - 16
        if limit <= 0:
            raise BadRequest(f"prompt of {len(toks)} tokens does not fit the context of {self.args.ctx}")
        max_new = limit if max_new <= 0 else min(max_new, limit)
        budget = body.get("thinking_budget", (body.get("chat_template_kwargs") or {}).get("thinking_budget",
                                                                                         self.args.think_budget))
        return Job(body, toks, thinking, max_new, int(budget or 0) if thinking else 0)

    def submit(self, job):
        self.queue.put(job)

    def prompt_start(self, toks):
        """brings the selected slot to a prefix of `toks` it already holds; returns that prefix's length"""
        eng = self.eng
        seq = eng.sequence()
        lcp = common_prefix(seq, toks)
        lcp = min(lcp, len(toks) - 1)  # at least one token is fed (it produces the next one)
        log(f"reuse: cached {len(seq)}, prompt {len(toks)}, common prefix {lcp}, "
            f"committed {eng.committed()}, checkpoint {eng.checkpoint_pos()}")
        if seq and lcp >= eng.committed():
            eng.truncate(lcp)
            return lcp
        ck = eng.checkpoint_pos()
        if 0 < ck <= lcp and eng.rewind_to_checkpoint():
            return ck
        eng.reset()
        return 0

    def sampling(self, body):
        """sets the selected slot's sampling from the request"""
        temp = float(body.get("temperature", 0.8) if body.get("temperature") is not None else 0.8)
        top_p = float(body.get("top_p", 0.95) if body.get("top_p") is not None else 0.95)
        top_k = int(body.get("top_k", 40) if body.get("top_k") is not None else 40)
        min_p = float(body.get("min_p", 0.05) if body.get("min_p") is not None else 0.05)
        seed = body.get("seed")
        seed = random.getrandbits(63) if seed is None or int(seed) < 0 else int(seed)
        if top_k <= 0 or top_k > TOPK_MAX:
            self.warn_once("top_k", f"top_k {top_k} -> {TOPK_MAX} (the megakernel samples from at most {TOPK_MAX} candidates)")
            top_k = TOPK_MAX
        for k in ("presence_penalty", "frequency_penalty"):
            if float(body.get(k) or 0) != 0:
                self.warn_once(k, f"{k} is not supported by the megakernel and is ignored")
        if float(body.get("repetition_penalty") or body.get("repeat_penalty") or 1.0) != 1.0:
            self.warn_once("rep", "repetition_penalty is not supported by the megakernel and is ignored")
        if temp <= 0:
            self.eng.set_sampling(0.0, 1.0, TOPK_MAX, 0.0, seed)
        else:
            self.eng.set_sampling(temp, top_p, top_k, min_p, seed)

    # ---- /status

    def track(self, s, **kw):
        """updates slot s's /status snapshot; a new dict each time, so a reader never sees it half-written"""
        self.slot_status[s] = {**self.slot_status[s], **kw, "updated": time.time()}

    def status(self):
        """the most recently updated busy slot at the top level (one-slot layout), every slot under `slots`"""
        slots = list(self.slot_status)
        busy = [st for st in slots if st["busy"]]
        top = max(busy or slots, key=lambda st: st["updated"])
        return {**top, "busy": bool(busy), "last": self.last, "queued": self.queue.qsize(), "slots": slots}

    @staticmethod
    def rate(n_gen, elapsed):
        """decode tok/s: the first token comes out of the prefill, so it is not part of the decode time"""
        return (n_gen - 1) / elapsed if n_gen > 1 and elapsed > 0 else 0.0

    def live(self, job, elapsed):
        """/status fields while tokens are generated: the phase and the tail of what it is producing"""
        if job.parser.mode == "tools":
            phase, text = "tool_call", job.parser.tool_text[-STATUS_TAIL:]
        elif job.parser.mode == "reasoning":
            phase, text = "thinking", job.tail["reasoning"]
        else:
            phase, text = "writing", job.tail["content"]
        return {"phase": phase, "gen": job.n_gen, "tok_s": self.rate(job.n_gen, elapsed),
                "action": {"kind": phase, "text": text}}

    # ---- scheduler thread: the only one that drives the engine

    def run(self):
        while True:
            if not self.active:
                self.admit(self.queue.get())
            while len(self.active) < self.args.slots:
                try:
                    job = self.queue.get_nowait()
                except queue.Empty:
                    break
                self.admit(job)
            filling = [j for j in self.active.values() if j.fed < len(j.toks)]
            decoding = len(self.active) > len(filling)
            for job in filling:
                self.prefill(job, decoding)
            self.cycle()

    def pick_slot(self, toks):
        """the free slot holding the longest prefix of toks (of at least REUSE_MIN), else the least recently used"""
        free = [s for s in range(self.args.slots) if s not in self.active]

        def reuse(s):
            n = common_prefix(self.slot_seq[s], toks)
            return n if n >= REUSE_MIN else 0

        return max(free, key=lambda s: (reuse(s), -self.slot_used[s]))

    def admit(self, job):
        """assigns a slot and the reusable prefix; the rest of the prompt is fed by prefill()"""
        if job.cancelled:
            job.events.put(("done", ("length", [], {}, {})))
            return
        s = job.slot = self.pick_slot(job.toks)
        self.active[s] = job
        self.track(s, busy=True, phase="prefill", gen=0, tok_s=0.0, ctx=len(job.toks), started=time.time(), prefill_s=None,
                   prefill_n=None)
        try:
            self.eng.select(s)
            self.sampling(job.body)
            job.t0 = time.time()
            job.reused = job.fed = self.prompt_start(job.toks)
        except Exception as e:
            self.fail(job, e)

    def prefill(self, job, chunked):
        """feeds the job's prompt, one chunk of it while other jobs are decoding (they stall one chunk at most)"""
        try:
            self.eng.select(job.slot)
            rest = job.toks[job.fed:]
            if chunked and len(rest) > self.args.prefill_chunk:
                # an intermediate chunk ends with a generated token, which the next chunk's position overwrites
                self.eng.append(rest[:self.args.prefill_chunk])
                job.fed += self.args.prefill_chunk
                self.eng.truncate(job.fed)
                self.track(job.slot, prefilled=job.fed)
                return
            tok = self.eng.prefill(rest)
            job.fed = len(job.toks)
        except Exception as e:
            self.fail(job, e)
            return
        job.t1 = time.time()
        self.track(job.slot, prefill_s=job.t1 - job.t0, prefill_n=len(job.toks) - job.reused)
        self.consume(job, [tok])

    def cycle(self):
        """one decode cycle of every active job: forced thinking stops one by one, the rest in one batched step"""
        stepping = []
        for s, job in list(self.active.items()):
            if job.cancelled:
                self.complete(job)
            elif job.fed < len(job.toks):
                continue
            elif job.budget and job.n_think >= job.budget and job.parser.mode == "reasoning":
                try:
                    self.eng.select(s)
                    tok = self.eng.append(self.eng.tokenize(THINK_STOP))
                except Exception as e:
                    self.fail(job, e)
                    continue
                for kind, delta in job.parser.feed(THINK_STOP):
                    job.show(kind, delta)
                log(f"slot {s}: thinking budget of {job.budget} tokens reached")
                job.budget = 0
                self.consume(job, [tok])
            else:
                stepping.append(s)
        if not stepping:
            return
        try:
            outs = self.eng.step_batch(stepping)
        except Exception as e:
            for s in stepping:
                self.fail(self.active[s], e)
            return
        for s, toks in zip(stepping, outs):
            self.consume(self.active[s], toks)

    def consume(self, job, toks):
        """feeds a cycle's tokens to the job; tokens after its end stay in the slot's sequence (truncated on reuse)"""
        done = False
        for t in toks:
            if job.n_gen >= job.max_new:
                break
            job.n_gen += 1
            if job.parser.mode == "reasoning":
                job.n_think += 1
            if self.eng.is_eog(t):
                job.finish, done = "stop", True
                break
            for kind, delta in job.parser.feed(job.dec.decode(self.eng.piece(t))):
                job.show(kind, delta)
        if time.time() - job.shown >= 0.25:
            job.shown = time.time()
            self.track(job.slot, **self.live(job, job.shown - job.t1))
        if done or job.n_gen >= job.max_new:
            self.complete(job)

    def release(self, job):
        s = job.slot
        del self.active[s]
        self.eng.select(s)
        self.slot_seq[s] = self.eng.sequence()
        self.slot_used[s] = time.time()

    def complete(self, job):
        for kind, delta in job.parser.feed(job.dec.decode(b"", final=True), final=True):
            job.show(kind, delta)
        calls = parse_tool_calls(job.parser.tool_text, job.tools) if job.parser.tool_text else []
        if calls:
            job.finish = "tool_calls"
        t2 = time.time()
        n_gen, n_prompt = job.n_gen, len(job.toks)
        end = "cancelled" if job.cancelled else job.finish
        log(f"slot {job.slot}: prompt {n_prompt} tok ({job.reused} reused) in {job.t1 - job.t0:.2f} s, {n_gen} tok "
            f"in {t2 - job.t1:.2f} s ({self.rate(n_gen, t2 - job.t1):.1f} tok/s), "
            f"{end}")
        usage = {"prompt_tokens": n_prompt, "completion_tokens": n_gen, "total_tokens": n_prompt + n_gen,
                 "prompt_tokens_details": {"cached_tokens": job.reused}}
        timings = {"prompt_n": n_prompt - job.reused, "prompt_ms": (job.t1 - job.t0) * 1e3, "predicted_n": n_gen,
                   "predicted_ms": (t2 - job.t1) * 1e3, "predicted_per_second": self.rate(n_gen, t2 - job.t1)}
        action = self.live(job, t2 - job.t1)["action"]
        if calls:
            action = {"kind": "tool_call", "calls": [
                {"name": c["function"]["name"],
                 "arguments": {k: v[:STATUS_TAIL] if isinstance(v, str) else v
                               for k, v in json.loads(c["function"]["arguments"]).items()}} for c in calls]}
        if n_gen > 1:  # a 1-token request (warm-up, title) has no decode speed: keep the last real job's figures
            self.last = {"slot": job.slot, "tok_s": timings["predicted_per_second"], "gen": n_gen, "ctx": n_prompt,
                         "reused": job.reused, "prefill_s": job.t1 - job.t0, "finish": end, "at": t2}
        self.track(job.slot, busy=False, phase="idle", action=action)
        self.release(job)
        job.events.put(("done", (job.finish, calls, usage, timings)))

    def fail(self, job, e):
        """engine error: the job reports it, its slot starts empty"""
        log(f"slot {job.slot} error:", repr(e))
        s = job.slot
        self.active.pop(s, None)
        try:
            self.eng.select(s)
            self.eng.reset()
        except Exception as e2:
            log("reset after error failed:", repr(e2))
        self.slot_seq[s] = []
        self.track(s, busy=False, phase="idle")
        job.events.put(("error", e))


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def make_handler(srv):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/health", "/v1/health"):
                self._json(200, {"status": "ok"})
            elif path in ("/v1/models", "/models"):
                self._json(200, {"object": "list", "data": [{"id": srv.model_id, "object": "model", "owned_by": "openjet"}]})
            elif path == "/status":
                self._json(200, srv.status())
            elif path == "/props":
                self._json(200, {"default_generation_settings": {"n_ctx": srv.args.ctx}, "total_slots": srv.args.slots})
            else:
                self._json(404, {"error": {"message": "not found", "type": "not_found_error"}})

        def do_POST(self):
            path = self.path.split("?")[0]
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            if path not in ("/v1/chat/completions", "/chat/completions"):
                self._json(404 if not path.startswith("/slots") else 501,
                           {"error": {"message": f"{path} is not supported by the megakernel server", "type": "not_supported"}})
                return
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                self._json(400, {"error": {"message": "invalid JSON", "type": "invalid_request_error"}})
                return
            try:
                job = srv.prepare(body)
            except BadRequest as e:
                self._json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
                return
            srv.submit(job)
            try:
                if body.get("stream"):
                    self._stream(job)
                else:
                    self._complete(job)
            except (BrokenPipeError, ConnectionResetError):
                job.cancelled = True
            except Exception as e:  # engine failure: report it, keep serving
                try:
                    self._json(500, {"error": {"message": str(e), "type": "server_error"}})
                except Exception:
                    pass

        def _complete(self, job):
            parts = {"reasoning": [], "content": []}
            finish, calls, usage, timings = job.wait(lambda k, t: parts[k].append(t))
            msg = {"role": "assistant", "content": "".join(parts["content"]) or (None if calls else "")}
            if parts["reasoning"]:
                msg["reasoning_content"] = "".join(parts["reasoning"]).strip()
            if calls:
                msg["tool_calls"] = calls
            self._json(200, {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
                             "model": srv.model_id, "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                             "usage": usage, "timings": timings})

        def _stream(self, job):
            cid, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
            started = [False]

            def chunk(delta, finish=None, usage=None, timings=None):
                obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": srv.model_id,
                       "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if usage is not None:
                    obj["usage"] = usage
                if timings is not None:
                    obj["timings"] = timings
                data = b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"
                self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
                self.wfile.flush()

            def head():
                if not started[0]:
                    started[0] = True
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    chunk({"role": "assistant", "content": None})

            def emit(kind, text):
                try:
                    head()
                    chunk({"reasoning_content": text} if kind == "reasoning" else {"content": text})
                except (BrokenPipeError, ConnectionResetError):
                    return False
                return True

            finish, calls, usage, timings = job.wait(emit)
            head()
            for i, c in enumerate(calls):
                chunk({"tool_calls": [{"index": i, **c}]})
            chunk({}, finish=finish, timings=timings)
            if (job.body.get("stream_options") or {}).get("include_usage"):
                chunk(None, usage=usage)
            data = b"data: [DONE]\n\n"
            self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n0\r\n\r\n")
            self.wfile.flush()

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("-c", "--ctx", type=int, default=32768)
    ap.add_argument("-d", "--draft", type=int, default=4)
    ap.add_argument("--draft-vocab", default=os.path.join(os.path.dirname(HERE), "draft_vocab_32k.bin"))
    ap.add_argument("--lib", default=os.path.join(os.path.dirname(HERE), "libqwen38mk.so"),
                    help="engine library; libqwen38mk_tp2.so runs tensor-parallel on two GPUs (MK_DEVICES=a,b)")
    ap.add_argument("--kv", choices=["auto", "f16", "q8"], default="auto",
                    help="KV cache format (auto: q8_0 only when the fp16 cache does not fit)")
    ap.add_argument("--slots", type=int, default=1,
                    help="requests decoded at once (1..8), each with its own KV cache of -c tokens; with the TP2 "
                         "library they share one batched cycle")
    ap.add_argument("--prefill-chunk", type=int, default=2048,
                    help="prompt tokens fed per step while other requests decode (they stall one chunk at most)")
    ap.add_argument("--think-budget", type=int, default=0,
                    help="reasoning tokens per request before </think> is forced (0: unlimited); "
                         "a request's thinking_budget overrides it")
    args = ap.parse_args()
    srv = Server(args)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(srv))
    httpd.daemon_threads = True
    log(f"listening on http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
