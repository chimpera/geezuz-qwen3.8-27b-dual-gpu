import sys
import os, sys, random, collections, hashlib
sys_path_dir = os.path.dirname(os.path.abspath(__file__))
sys.argv=[sys.argv[0]]; sys.path.insert(0, sys_path_dir)
import sched_sim as sim
mk = sim.mk
def nxt(self, s):
    h = int(hashlib.md5(str((len(s["toks"]), s["toks"][-3:])).encode()).hexdigest(), 16)
    return sim.EOG if h % 600 == 0 else 1 + h % 799
sim.FakeEng._next = nxt
# step: deterministic number of outputs per cycle would differ by batch; make output count depend on position only
orig = sim.FakeEng.step_batch
def step_batch(self, ids):
    need = 0
    for i in ids:
        s = self.seqs[i]; n = len(s["toks"]) - 1
        if n + sim.KV_AHEAD > self.ctx: raise RuntimeError("context full")
        need += max(0, -(-(n + sim.KV_AHEAD) // sim.NP) - s["blocks"])
    if need and not self.room(need): raise mk.PoolFull("full")
    outs = []
    for i in ids:
        s = self.seqs[i]; n = len(s["toks"]) - 1
        s["blocks"] = max(s["blocks"], -(-(n + sim.KV_AHEAD) // sim.NP))
        o = []
        s["dn"] = len(s["toks"])  # engine: dn_c = n + 1, the first output's position
        for _ in range(1 + len(s["toks"]) % 3):
            t = nxt(self, s); s["toks"].append(t); o.append(t)
        outs.append(o)
    return outs
sim.FakeEng.step_batch = step_batch
mk.Engine  # noqa
def piece(self, t): return (str(t) + " ").encode()
sim.FakeEng.piece = piece
def texts(jobs):
    out = []
    for j in jobs:
        txt = "".join(e[2] for e in j.events.queue if e[0] == "delta")
        out.append((txt, j.n_gen, j.preempts, j.cancelled, j.budget, j.stopped))
    return out
diff = collections.Counter()
for seed in range(200):
    nb = random.Random(seed).choice([140, 200, 300])
    slots = random.Random(seed + 1).choice([2, 4, 8])
    sim.random.seed(0)
    r1, s1, j1 = sim.run(seed, nb, slots, 16)
    r2, s2, j2 = sim.run(seed, 100000, slots, 16)
    for a, b, ja in zip(texts(j1), texts(j2), j1):
        if ja.cancelled: continue
        key = ("pre" if a[2] else "nopre", "think" if a[5] else "", a[0] == b[0])
        diff[key] += 1
        if a[0] != b[0] and diff[key] < 3:
            # find first difference
            i = next(k for k in range(min(len(a[0]), len(b[0]))+1) if k >= min(len(a[0]), len(b[0])) or a[0][k] != b[0][k])
            print("DIFF", key, "n_gen", a[1], b[1], "...", a[0][max(0,i-60):i+60].replace("\n","|"), "<<>>", b[0][max(0,i-60):i+60].replace("\n","|"))
print(diff)
