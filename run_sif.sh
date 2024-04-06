#!/bin/bash

# Mount options
ROOT_DIR="$(pwd)"

# Singularity options
CONTAINER_OR_SANDBOX="/h/mchoi/scratch/ngc/pytorch:24.02-py3.sif"

module load singularity-ce/3.8.2

singularity run \
	--nv \
	--cleanenv \
	--no-home \
    --writable-tmpfs \
	--env "TORCH_CUDA_ARCH_LIST=8.0" \
	-B "${ROOT_DIR}:/mnt" \
	"${CONTAINER_OR_SANDBOX}" \
	bash
