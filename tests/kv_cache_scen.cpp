// Deterministic adversarial scenarios on top of the fuzz oracle (fuzz.cpp included, its main renamed).
#define main fuzz_main
#include "kv_cache_fuzz.cpp"
#undef main
static Seq open_prompt(const std::vector<int> & p) {
    Seq q; auto o = kc->open(p.data(), (int) p.size(), true, q.blk); q.lane = o.lane;
    if (o.pos) { q.toks.assign(p.begin(), p.begin() + o.pos); q.c = q.valid = o.pos; verify(q); }
    else dev->lane[q.lane] = {0, H({}, 0)};
    seqs.push_back(q); return seqs.back();
}
int main() {
    rng.seed(5);
    // 1: shared partial block, two sequences append to it
    NB = 40; NP = 4; SNB = 2; LANES = 3;
    dev = new Fake; kc = new KvCache<Meta>(NB, NP, NBL, SNB, LANES, dev);
    std::vector<int> sys = {0, 1, 2, 0, 1, 2}; // 6 tokens: snapshot at 6 lands mid-block 1
    open_prompt({}); feed(seqs[0], {0,1,2,0,1,2,2,2}, true, {6});
    kc->close(seqs[0].lane, seqs[0].toks.data(), seqs[0].c, seqs[0].blk, true, Meta{seqs[0].c}); seqs.clear(); check_all();
    std::vector<int> pa = sys; pa.push_back(1); pa.push_back(1);
    std::vector<int> pb = sys; pb.push_back(0); pb.push_back(0);
    open_prompt(pa); open_prompt(pb);
    if (seqs[0].c != 6 || seqs[1].c != 6) FAIL("expected reuse at 6, got %d %d", seqs[0].c, seqs[1].c);
    if (seqs[0].blk[1] != seqs[1].blk[1]) FAIL("partial block not shared");
    feed(seqs[0], {1, 1}, true, {}); check_all();
    feed(seqs[1], {0, 0}, true, {}); check_all();
    for (int k = 0; k < 6; k++) { op_step(); check_all(); }
    puts("scenario 1 (shared partial block appended by two sequences): ok");
    // 2: evict everything while open sequences reference cached blocks: fill pool via a long sequence
    { Seq & q = seqs[0]; std::vector<int> big(30, 2); feed(q, big, true, {}); check_all(); }
    int full = 0;
    for (int k = 0; k < 40; k++) { size_t b = seqs[1].blk.size(); op_step(); check_all(); if (seqs[1].blk.size() == b) full++; }
    printf("scenario 2 (eviction under pressure with live sharers): ok, free %d cached %d\n", kc->free_blocks(), kc->cached_blocks());
    // 3: pool exhaustion mid-ensure: drain free list with a third seq, then ensure_all must be all-or-nothing
    open_prompt({}); feed(seqs[2], std::vector<int>(20, 1), true, {}); check_all();
    for (int k = 0; k < 30; k++) { op_step(); check_all(); }
    printf("scenario 3 (exhaustion, ensure_all all-or-nothing): ok, free %d\n", kc->free_blocks());
}
