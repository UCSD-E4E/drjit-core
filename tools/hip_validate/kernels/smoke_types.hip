// Every type spelling type_name_hip (src/hip_eval.h) can emit must exist on
// BOTH arms. Without this, a table entry that is valid for gfx90a but missing
// from the NVRTC prelude silently costs us the execution arm for any kernel
// touching that type -- which is how int32_t was found missing.
//
// Keep in step with tests/hip_types.cpp: that asserts the spellings, this
// asserts they resolve.
DRJIT_KERNEL void k(float *out, const float *in, unsigned n) {
    uint32_t i = DRJIT_TID;
    if (i >= n) return;

    bool       b   = in[i] > 0.5f;
    int8_t     i8  = (int8_t)   1;
    uint8_t    u8  = (uint8_t)  2;
    int16_t    i16 = (int16_t)  3;
    uint16_t   u16 = (uint16_t) 4;
    int32_t    i32 = (int32_t)  5;
    uint32_t   u32 = (uint32_t) 6;
    int64_t    i64 = (int64_t)  7;
    uint64_t   u64 = (uint64_t) 8;
    float      f32 = 9.0f;
    double     f64 = 10.0;          // full-rate on gfx90a; must not be promoted away
    DRJIT_HALF f16 = DRJIT_HALF_FROM_BITS(0x3c00u);  // 1.0

    // Pointer is spelled uint64_t by the table; round-trip through it.
    uint64_t   ptr = (uint64_t) (unsigned long long) 0x1000ull;

    out[i] = (float) b + (float) i8 + (float) u8 + (float) i16 + (float) u16 +
             (float) i32 + (float) u32 + (float) i64 + (float) u64 +
             f32 + (float) f64 + (float) f16 + (float) (ptr >> 12);
}
