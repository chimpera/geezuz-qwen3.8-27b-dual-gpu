# Build image for the megakernel: the CUDA devel image plus cmake and git,
# which it lacks. No GPU is needed to build (the Makefile falls back to sm_86).
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

RUN apt-get update \
	&& apt-get install -y --no-install-recommends ca-certificates cmake git \
	&& rm -rf /var/lib/apt/lists/*
