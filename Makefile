LLAMA ?= ../llama.cpp
NVCC ?= /usr/local/cuda/bin/nvcc
# the first GPU's architecture (sm_86 RTX 30, sm_89 RTX 40, sm_120 RTX 50); sm_86 if nvidia-smi is missing
ARCH ?= $(or $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. '),86)
ARCH := sm_$(patsubst sm_%,%,$(ARCH))
NVFLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xptxas -dlcm=cg -Xptxas -v -Xcompiler -fPIC
LLAMA_BIN ?= $(LLAMA)/build/bin
# binaries look for libllama next to themselves first (release tarball); a release build leaves out the build path
RPATH = $(if $(RELEASE),$$ORIGIN,$$ORIGIN:$(abspath $(LLAMA_BIN)))
LDLLAMA = -L$(LLAMA_BIN) -lllama -lggml-base -Xlinker -rpath -Xlinker '$(RPATH)'
VERSION ?= v0.2.0
DIST = geezuz-$(VERSION)-$(ARCH)

all: geezuz libgeezuz.so geezuz_tp2 libgeezuz_tp2.so tools/vocab_freq tools/llama_ref

# ptxas register / spill report for every phase function goes to ptxas.log
qwen38_mk.o: qwen38_mk.cu qwen38_mk.h kv_cache.h gguf.h prefill.cuh prefill_gemm.cuh batch.cuh mma.cuh q8_tile.h
	$(NVCC) $(NVFLAGS) -I$(LLAMA)/ggml/include -c qwen38_mk.cu -o $@ 2> ptxas.log || (cat ptxas.log; false)

geezuz: qwen38_mk.o main.cu logit_stats.h
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -I$(LLAMA)/include -I$(LLAMA)/ggml/include main.cu qwen38_mk.o \
		$(LDLLAMA) -o $@

# engine + tokenizer as a shared library with a C API (capi.cpp), used by server/mk_server.py
libgeezuz.so: qwen38_mk.o capi.cpp qwen38_mk.h
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -shared -Xcompiler -fPIC -I$(LLAMA)/include -I$(LLAMA)/ggml/include capi.cpp qwen38_mk.o \
		$(LDLLAMA) -o $@

# TP=2 variant: same compile with MK_TP=2, its own ptxas report
qwen38_mk_tp2.o: qwen38_mk.cu qwen38_mk.h kv_cache.h gguf.h prefill.cuh prefill_gemm.cuh batch.cuh mma.cuh q8_tile.h
	$(NVCC) $(NVFLAGS) -DMK_TP=2 -I$(LLAMA)/ggml/include -c qwen38_mk.cu -o $@ 2> ptxas_tp2.log || (cat ptxas_tp2.log; false)

geezuz_tp2: qwen38_mk_tp2.o main.cu logit_stats.h
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -I$(LLAMA)/include -I$(LLAMA)/ggml/include main.cu qwen38_mk_tp2.o \
		$(LDLLAMA) -o $@

libgeezuz_tp2.so: qwen38_mk_tp2.o capi.cpp qwen38_mk.h
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -shared -Xcompiler -fPIC -I$(LLAMA)/include -I$(LLAMA)/ggml/include capi.cpp qwen38_mk_tp2.o \
		$(LDLLAMA) -o $@

tools/vocab_freq: tools/vocab_freq.cpp
	g++ -O2 -std=c++17 -I$(LLAMA)/include -I$(LLAMA)/ggml/include $< -L$(LLAMA_BIN) -lllama \
		-Wl,-rpath,$(abspath $(LLAMA_BIN)) -o $@

# llama.cpp's next-token distributions for the accuracy check (see "Correctness" in README.md)
tools/llama_ref: tools/llama_ref.cpp logit_stats.h
	g++ -O2 -std=c++17 -pthread -I$(LLAMA)/include -I$(LLAMA)/ggml/include $< -L$(LLAMA_BIN) -lllama \
		-Wl,-rpath,$(abspath $(LLAMA_BIN)) -o $@

# release tarball: the four binaries relinked without the build path, libllama and its ggml libraries
# (build llama.cpp with -DGGML_NATIVE=OFF so they run on any x86-64 CPU), the server and the docs
release: qwen38_mk.o qwen38_mk_tp2.o
	rm -f geezuz libgeezuz.so geezuz_tp2 libgeezuz_tp2.so
	$(MAKE) RELEASE=1 geezuz libgeezuz.so geezuz_tp2 libgeezuz_tp2.so
	rm -rf dist/$(DIST) && mkdir -p dist/$(DIST)/server
	cp geezuz libgeezuz.so geezuz_tp2 libgeezuz_tp2.so draft_vocab_32k.bin LICENSE README.md CHANGELOG.md dist/$(DIST)/
	cp server/mk_server.py dist/$(DIST)/server/
	cp $(LLAMA)/LICENSE dist/$(DIST)/LICENSE.llama.cpp
	LD_LIBRARY_PATH=$(LLAMA_BIN) ldd libgeezuz_tp2.so | awk '/=> .*(llama|ggml)/ {print $$3}' | xargs -I{} cp -L {} dist/$(DIST)/
	cd dist && tar czf $(DIST).tar.gz $(DIST) && sha256sum $(DIST).tar.gz > $(DIST).tar.gz.sha256

.PHONY: all release clean
clean:
	rm -f geezuz libgeezuz.so qwen38_mk.o ptxas.log geezuz_tp2 libgeezuz_tp2.so qwen38_mk_tp2.o ptxas_tp2.log \
		tools/vocab_freq tools/llama_ref
	rm -rf dist
