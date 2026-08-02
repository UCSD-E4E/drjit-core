# Target of the precompiled PTX files shipped with Dr.Jit-core:
#   - CUDA 12.2 (nvcc 12.2.140)
#   - PTX ISA version 8.2
#   - Minimum CUDA driver: 535 (the one bundled with CUDA 12.2)
#   - Compute Capability >= 7.5 (Turing and newer)

COMPUTE_CAPABILITY=compute_75
CUDA_VER=12.2
HOST_CC=/usr/bin/gcc-12
NVCC=/usr/local/cuda-$(CUDA_VER)/bin/nvcc -ccbin $(HOST_CC) -m64 --ptx --expt-relaxed-constexpr -std=c++14 -Xcudafe --diag_suppress=550 -Xcudafe --diag_suppress=177 -Xcudafe --diag_suppress=2912 -Xcudafe --display_error_number

all: kernels_75.lz4

kernels_75.ptx: common.h block_reduce.cuh block_prefix_reduce.cuh reduce_2.cuh compress.cuh mkperm.cuh misc.cuh gemm.cuh kernels.cu
	$(NVCC) -Wno-deprecated-gpu-targets -gencode arch=$(COMPUTE_CAPABILITY),code=$(COMPUTE_CAPABILITY) kernels.cu -o kernels_75.ptx

#kernels.dict:
#	zstd --train train/* --maxdict=65536 -o kernels.dict

pack: pack.c
	$(CC) pack.c ../ext/lz4/lz4hc.c ../ext/lz4/lz4.c ../ext/lz4/xxhash.c -o pack -I ../ext/lz4

kernels_75.lz4: pack kernels_75.ptx
	./pack

# Precompiled AMD kernels shipped with Dr.Jit-core (PLAN.md Phase 3). Requires
# ROCm, unlike the build itself, which only embeds the committed blob.
#
# Two steps rather than one: --genco emits a clang OFFLOAD BUNDLE, and we embed
# the code object it contains. See pack_hip.c for why.
HIP_ARCH ?= gfx90a
HIPCC ?= hipcc
BUNDLER ?= clang-offload-bundler

hip: kernels_$(HIP_ARCH).lz4

kernels_$(HIP_ARCH).co: common.h block_reduce.cuh block_prefix_reduce.cuh reduce_2.cuh compress.cuh mkperm.cuh misc.cuh kernels.cu
	$(HIPCC) --offload-arch=$(HIP_ARCH) --genco -std=c++17 -O3 kernels.cu -o $@

kernels_$(HIP_ARCH).elf: kernels_$(HIP_ARCH).co
	$(BUNDLER) --unbundle --type=o \
		--targets=hipv4-amdgcn-amd-amdhsa--$(HIP_ARCH) \
		--input=$< --output=$@

pack_hip: pack_hip.c
	$(CC) pack_hip.c ../ext/lz4/lz4hc.c ../ext/lz4/lz4.c -o pack_hip -I ../ext/lz4

kernels_$(HIP_ARCH).lz4: pack_hip kernels_$(HIP_ARCH).elf
	./pack_hip $(HIP_ARCH)

# Precompiled Metal kernels shipped with Dr.Jit-core (macOS only). Requires the
# full Metal toolchain (xcrun metal), unlike the build itself, which only embeds
# the committed metallib.
metal: metal_kernels.metallib

metal_kernels.metallib: metal_kernels.metal
	xcrun -sdk macosx metal -mmacosx-version-min=13.0 -std=metal3.2 -fno-fast-math -c metal_kernels.metal -o metal_kernels.air
	xcrun -sdk macosx metallib metal_kernels.air -o metal_kernels.metallib
	rm -f metal_kernels.air

clean:
	rm -f kernels_75.ptx kernels.bin metal_kernels.air \
	      kernels_$(HIP_ARCH).co kernels_$(HIP_ARCH).elf
