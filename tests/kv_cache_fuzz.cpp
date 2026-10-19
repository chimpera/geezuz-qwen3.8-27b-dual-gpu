// Content-oracle fuzz of kv_cache.h: a fake device stores per-row content hashes and per-block snapshot records,
// so any write into a shared block, missing CoW, missing table upload, freed-while-referenced block or wrong
// snapshot restore shows up as a content mismatch.
#define private public
#include "kv_cache.h"
#undef private
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>

static int NP = 4, NBL = 16, SNB = 2, LANES = 3, NB = 40, ALPH = 3;
static const uint64_t GARB = 1ULL << 63;
static std::mt19937_64 rng;
static int R(int n) { return (int) (rng() % (uint64_t) n); }

static uint64_t H(const std::vector<int> & t, int n) { // hash of t[0, n)
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) h = (h ^ (uint64_t) (t[i] + 7)) * 1099511628211ULL;
    return h & ~GARB;
}

#define FAIL(...) do { fprintf(stderr, "FAIL line %d: ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); abort(); } while (0)

struct Meta { int pos = -1; };

struct Fake : KvDev {
    std::vector<std::vector<uint64_t>> data;
    struct SR { int pos = -1; uint64_t h = 0; int idx = -1; };
    std::vector<SR> snap;
    std::vector<std::pair<int, uint64_t>> lane;
    std::vector<std::vector<int>> bt;
    Fake() : data(NB, std::vector<uint64_t>(NP, GARB | 1)), snap(NB), lane(LANES, {0, H({}, 0)}), bt(LANES, std::vector<int>(NBL, 0)) {}
    void copy_block(int s, int d) override { data[d] = data[s]; snap[d] = snap[s]; }
    void snap_save(int l, int pos, const std::vector<int> & b) override {
        if (lane[l].first != pos) FAIL("snap_save lane %d at %d holds %d", l, pos, lane[l].first);
        for (size_t i = 0; i < b.size(); i++) snap[b[i]] = {pos, lane[l].second, (int) i};
    }
    void snap_load(int l, int pos, const std::vector<int> & b) override {
        for (size_t i = 0; i < b.size(); i++)
            if (snap[b[i]].pos != pos || snap[b[i]].idx != (int) i || snap[b[i]].h != snap[b[0]].h) FAIL("snap_load blk %d corrupt", b[i]);
        lane[l] = {pos, snap[b[0]].h};
    }
    void upload_bt(int l, int i0, const int * ids, int k) override {
        if (i0 < 0 || i0 + k > NBL) FAIL("upload range");
        for (int i = 0; i < k; i++) bt[l][i0 + i] = ids[i];
    }
};

struct Seq {
    std::vector<int> toks, blk;
    int lane = -1, c = 0, valid = 0, from = 0, upto = 0; // c: committed (lane state), valid rows [0, valid)
    bool dirty = false; // truncated: next op must be a feed
};

static Fake * dev;
static KvCache<Meta> * kc;
static std::vector<Seq> seqs; // open
static std::vector<std::vector<int>> history;
static long n_pf = 0, n_hit = 0, n_ops = 0, n_snap = 0;

static void write_row(Seq & q, int p, bool real) {
    if (p < q.from || p >= q.upto) FAIL("write %d outside ensured [%d,%d)", p, q.from, q.upto);
    const int b = dev->bt[q.lane][p / NP];
    if ((size_t) (p / NP) >= q.blk.size() || b != q.blk[p / NP]) FAIL("device table stale at %d", p / NP);
    if (kc->ref[b] != 1) FAIL("write into shared block %d (ref %d) at pos %d", b, kc->ref[b], p);
    dev->data[b][p % NP] = real ? H(q.toks, p + 1) : (GARB | rng());
}
static void verify(const Seq & q) {
    for (int p = 0; p < q.valid; p++) {
        const int b = dev->bt[q.lane][p / NP];
        if (b != q.blk[p / NP]) FAIL("table mismatch");
        if (dev->data[b][p % NP] != H(q.toks, p + 1)) FAIL("row %d of seq (lane %d) wrong", p, q.lane);
    }
    if (dev->lane[q.lane].first != q.c || dev->lane[q.lane].second != H(q.toks, q.c)) FAIL("lane state wrong");
}
// every indexed node's block content and every stored snapshot record against the tokens of its path
static void verify_cache() {
    for (auto & kv : kc->nodes) {
        if (kv.first == 0) continue;
        std::vector<int> path;
        for (int n = kv.first; n > 0; n = kc->nodes.at(n).parent) path.push_back(n);
        std::vector<int> t;
        for (int i = (int) path.size() - 1; i >= 0; i--) {
            const auto & nd = kc->nodes.at(path[i]);
            t.insert(t.end(), nd.toks.begin(), nd.toks.end());
        }
        const auto & nd = kv.second;
        for (int i = 0; i < (int) nd.toks.size(); i++)
            if (dev->data[nd.blk][i] != H(t, nd.start + i + 1)) FAIL("cached node %d row %d wrong", kv.first, i);
    }
    for (auto & kv : kc->snaps) {
        const auto & s = kv.second;
        std::vector<int> t;
        std::vector<int> path;
        for (int n = s.node; n > 0; n = kc->nodes.at(n).parent) path.push_back(n);
        for (int i = (int) path.size() - 1; i >= 0; i--) {
            const auto & nd = kc->nodes.at(path[i]);
            t.insert(t.end(), nd.toks.begin(), nd.toks.end());
        }
        if (s.meta.pos != s.pos) FAIL("meta");
        if (s.lane >= 0) {
            if (dev->lane[s.lane].first != s.pos || dev->lane[s.lane].second != H(t, s.pos)) FAIL("parked lane state");
        } else {
            for (size_t i = 0; i < s.blks.size(); i++) {
                const auto & r = dev->snap[s.blks[i]];
                if (r.pos != s.pos || r.idx != (int) i || r.h != H(t, s.pos)) FAIL("snapshot %d blocks corrupt", kv.first);
            }
        }
    }
}
static void check_all() {
    std::vector<const std::vector<int> *> v;
    for (auto & q : seqs) v.push_back(&q.blk);
    kc->check(v);
    for (auto & q : seqs) verify(q);
    verify_cache();
}

static std::vector<int> rand_toks(int n) {
    std::vector<int> t(n);
    for (int & x : t) x = R(ALPH);
    return t;
}

static bool do_ensure(Seq & q, int from, int upto) {
    auto blk0 = q.blk;
    auto fr0 = kc->free_.size();
    (void) fr0;
    try {
        kc->ensure(q.blk, q.lane, from, upto);
    } catch (const MKPoolFull &) {
        if (q.blk != blk0) FAIL("ensure changed blocks on PoolFull");
        return false;
    }
    q.from = from, q.upto = upto;
    return true;
}

static void feed(Seq & q, const std::vector<int> & add, bool snap_end, const std::vector<int> & hints) {
    const int len = (int) (q.toks.size() + add.size());
    if (len + 9 > NBL * NP) return;
    if (!do_ensure(q, q.c, len + 5)) return;
    q.toks.insert(q.toks.end(), add.begin(), add.end());
    int pos = q.c;
    std::vector<int> cuts;
    for (int h : hints)
        if (h >= std::max(pos, 1) && h < len - 1) cuts.push_back(h);
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    if (len - pos >= 2) {
        for (int h : cuts) {
            for (int p = pos; p < h; p++) write_row(q, p, true);
            pos = q.c = h;
            dev->lane[q.lane] = {h, H(q.toks, h)};
            n_snap += kc->snap_take(q.toks.data(), h, q.blk, q.lane, Meta{h});
        }
        for (int p = pos; p < len - 1; p++) write_row(q, p, true);
        q.c = len - 1;
        dev->lane[q.lane] = {len - 1, H(q.toks, len - 1)};
        if (snap_end) n_snap += kc->snap_take(q.toks.data(), len - 1, q.blk, q.lane, Meta{len - 1});
        pos = len - 1;
    }
    for (int p = pos; p < len; p++) write_row(q, p, true);
    q.c = len;
    dev->lane[q.lane] = {len, H(q.toks, len)};
    for (int p = len; p < len + 4; p++) write_row(q, p, false);
    q.toks.push_back(R(ALPH));
    q.valid = len;
    q.dirty = false;
    if (snap_end && (int) q.toks.size() - 1 == len && false) {}
}

static void step_one(Seq & q) {
    const int n = (int) q.toks.size() - 1;
    for (int p = q.c; p <= n; p++) write_row(q, p, true);
    for (int p = n + 1; p < n + 9; p++) write_row(q, p, false);
    const int a = R(5);
    for (int i = 0; i <= a; i++) q.toks.push_back(R(ALPH));
    for (int p = n + 1; p <= n + a; p++) write_row(q, p, true);
    q.c = n + 1;
    dev->lane[q.lane] = {q.c, H(q.toks, q.c)};
    q.valid = n + a + 1;
}

static void op_step() {
    std::vector<int> idx;
    for (int i = 0; i < (int) seqs.size(); i++)
        if (!seqs[i].dirty && !seqs[i].toks.empty() && (int) seqs[i].toks.size() - 1 + 9 <= NBL * NP && R(2)) idx.push_back(i);
    if (idx.empty()) return;
    std::vector<KvCache<Meta>::Req> rs;
    std::vector<std::vector<int>> b0;
    for (int i : idx) {
        Seq & q = seqs[i];
        rs.push_back({&q.blk, q.lane, q.c, (int) q.toks.size() - 1 + 9});
        b0.push_back(q.blk);
    }
    try {
        kc->ensure_all(rs);
    } catch (const MKPoolFull &) {
        for (size_t k = 0; k < idx.size(); k++)
            if (seqs[idx[k]].blk != b0[k]) FAIL("ensure_all partial on PoolFull");
        return;
    }
    for (int i : idx) {
        seqs[i].from = seqs[i].c, seqs[i].upto = (int) seqs[i].toks.size() - 1 + 9;
        step_one(seqs[i]);
    }
}

static void op_open() {
    if ((int) seqs.size() >= LANES) return;
    std::vector<int> prompt;
    if (!history.empty() && R(4)) {
        const auto & h = history[R((int) history.size())];
        prompt.assign(h.begin(), h.begin() + R((int) h.size() + 1));
    }
    auto extra = rand_toks(R(10));
    prompt.insert(prompt.end(), extra.begin(), extra.end());
    if ((int) prompt.size() + 10 > NBL * NP) prompt.resize(NBL * NP - 10);
    Seq q;
    auto o = kc->open(prompt.data(), (int) prompt.size(), true, q.blk);
    q.lane = o.lane;
    for (auto & s : seqs)
        if (s.lane == q.lane) FAIL("lane %d given twice", q.lane);
    n_pf++;
    if (o.pos > 0) {
        n_hit++;
        if (o.pos > (int) prompt.size() - 1) FAIL("reuse covers the whole prompt");
        if (o.meta.pos != o.pos) FAIL("meta pos");
        q.toks.assign(prompt.begin(), prompt.begin() + o.pos);
        q.c = o.pos;
        q.valid = o.pos;
        verify(q);
    } else {
        dev->lane[q.lane] = {0, H({}, 0)};
    }
    seqs.push_back(q);
    Seq & s = seqs.back();
    std::vector<int> rest(prompt.begin() + o.pos, prompt.end());
    if (rest.empty()) rest = rand_toks(1);
    std::vector<int> hints;
    for (int k = R(3); k-- > 0;) hints.push_back(R((int) prompt.size() + 1));
    feed(s, rest, R(4) != 0, hints);
    if (s.toks.empty()) { // pool short: close it again
        kc->close(s.lane, s.toks.data(), 0, s.blk, false, Meta{0});
        seqs.pop_back();
    }
}

static void op_close() {
    if (seqs.empty()) return;
    const int i = R((int) seqs.size());
    Seq & q = seqs[i];
    const bool keep = R(4) != 0;
    const int pos = std::min(q.c, (int) q.toks.size());
    kc->close(q.lane, q.toks.data(), pos, q.blk, keep, Meta{pos});
    history.push_back(q.toks);
    if (history.size() > 30) history.erase(history.begin());
    seqs.erase(seqs.begin() + i);
}

static void op_append() {
    if (seqs.empty()) return;
    Seq & q = seqs[R((int) seqs.size())];
    if (q.toks.empty()) return;
    std::vector<int> hints;
    if (R(2)) hints.push_back((int) q.toks.size() + R(4));
    feed(q, rand_toks(1 + R(8)), R(2), hints);
}

static void op_truncate() {
    if (seqs.empty()) return;
    Seq & q = seqs[R((int) seqs.size())];
    if ((int) q.toks.size() <= q.c) return;
    const int len = q.c + R((int) q.toks.size() - q.c + 1);
    q.toks.resize(len);
    kc->trim(q.blk, len);
    q.valid = std::min(q.valid, len);
    q.valid = std::min(q.valid, (int) q.blk.size() * NP);
    q.dirty = true;
}

int main(int argc, char ** argv) {
    const long ops = argc > 1 ? atol(argv[1]) : 100000;
    const int seed = argc > 2 ? atoi(argv[2]) : 1;
    if (argc > 3) NB = atoi(argv[3]);
    if (argc > 4) NP = atoi(argv[4]);
    if (argc > 5) SNB = atoi(argv[5]);
    if (argc > 6) ALPH = atoi(argv[6]);
    rng.seed(seed);
    dev = new Fake;
    kc = new KvCache<Meta>(NB, NP, NBL, SNB, LANES, dev);
    for (n_ops = 0; n_ops < ops; n_ops++) {
        switch (R(10)) {
            case 0: case 1: op_open(); break;
            case 2: case 3: op_close(); break;
            case 4: case 5: case 6: op_step(); break;
            case 7: case 8: op_append(); break;
            case 9: op_truncate(); break;
        }
        check_all();
    }
    printf("ok seed %d NB %d NP %d SNB %d: opens %ld hits %ld snaps %ld free %d cached %d\n", seed, NB, NP, SNB, n_pf, n_hit, n_snap,
           kc->free_blocks(), kc->cached_blocks());
}
