# Build image for the megakernel: the CUDA devel image plus cmake and git,
# which it lacks. No GPU is needed to build (the Makefile falls back to sm_86).
# UBUNTU=22.04 for release builds: glibc 2.35, so the binaries also run on older distributions
ARG UBUNTU=24.04
FROM nvidia/cuda:12.9.1-devel-ubuntu${UBUNTU}

RUN apt-get update \
	&& apt-get install -y --no-install-recommends ca-certificates cmake git \
	&& rm -rf /var/lib/apt/lists/*
