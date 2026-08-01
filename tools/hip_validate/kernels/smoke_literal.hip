// Proves the exact literal forms jitc_hip_literal() emits are compilable and
// correct on BOTH arms. tests/hip_literal.cpp asserts the strings; this asserts
// they mean what we think.
//
// Values chosen so a wrong reinterpretation is visible rather than plausible:
//   __uint_as_float(0x3f800000u)          == 1.0f
//   __longlong_as_double(0x3ff0...ull)    == 1.0
//   DRJIT_HALF_FROM_BITS(0x3c00u)         == 1.0
// so out[] lands on in[] + 3 exactly. A truncated FP64 path or a mangled half
// would shift the sum.
DRJIT_KERNEL void k(float *out, const float *in, unsigned n) {
    uint32_t i = DRJIT_TID;
    if (i >= n) return;

    float      f32 = __uint_as_float(0x3f800000u);
    double     f64 = __longlong_as_double(0x3ff0000000000000ull);
    DRJIT_HALF f16 = DRJIT_HALF_FROM_BITS(0x3c00u);

    // Integer suffix forms, exercised for compilability and width.
    uint64_t big = 0xdeadbeef00000000ull;
    int32_t  neg = (int32_t) 0xffffffffu;

    float acc = in[i] + f32 + (float) f64 + (float) f16;
    acc += (float) (big >> 60);      // 0xd == 13
    acc += (float) neg;              // -1
    out[i] = acc;
}
