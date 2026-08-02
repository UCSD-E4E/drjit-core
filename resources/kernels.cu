/*
    kernels.cu -- Supplemental kernels used by Dr.JIT

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"

#include "compress.cuh"
#include "mkperm.cuh"
#include "misc.cuh"
#include "block_reduce.cuh"
#include "block_prefix_reduce.cuh"
#include "reduce_2.cuh"

#if !DRJIT_KERNELS_HIP
// GEMM is CUDA-only: it uses __grid_constant__, which has no HIP spelling, and
// the HIP backend routes batched matrix products to rocBLAS instead. Nothing
// Mitsuba variant we target reaches HIPThreadState::batched_gemm.
#include "gemm.cuh"
#endif
