typedef signed char        i8;
typedef unsigned char      u8;
typedef short              i16;
typedef unsigned short     u16;
typedef int                i32;
typedef unsigned int       u32;
typedef long long          i64;
typedef unsigned long long u64;
typedef float              f32;
typedef double             f64;

#if defined(__HIP__) && !defined(__CUDACC_RTC__)
typedef _Float16 f16;
#elif defined(DRJIT_SHIM_HAVE_FP16)
#include <cuda_fp16.h>
typedef __half f16;
#else
struct f16 { u16 x; };  /* fp16 unavailable -- set DRJIT_HIP_SHIM_CUDA_INCLUDE */
#endif

__device__ inline f16 drjit_half_from_bits(u32 h) {
#if defined(__HIP__) && !defined(__CUDACC_RTC__)
    union { u16 u; f16 h; } c; c.u = (u16) h; return c.h;
#elif defined(DRJIT_SHIM_HAVE_FP16)
    return __ushort_as_half((u16) h);
#else
    f16 r; r.x = (u16) h; return r;
#endif
}
#define DRJIT_HALF_FROM_BITS(bits) drjit_half_from_bits((u32) (bits))

__device__ inline u16 drjit_half_to_bits(f16 h) {
#if defined(__HIP__) && !defined(__CUDACC_RTC__)
    union { f16 h; u16 u; } c; c.h = h; return c.u;
#elif defined(DRJIT_SHIM_HAVE_FP16)
    return __half_as_ushort(h);
#else
    return h.x;
#endif
}


#include <hiprt/hiprt_device.h>

struct Params {
    u32 size;
    void *args[12];
};

extern "C" __global__ void drjit_a45e36f486810f4035ee3432de3a077d(Params params) {
    u32 r0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (r0 >= params.size)
        return;
u64 r4 = (u64) params.args[0];
u64 r5 = (u64) params.args[1];
u64 r6 = (u64) params.args[2];
u32 r7 = *(const u32 *) params.args[3];
f32 r8 = __uint_as_float(0xbe99999au);
f32 r9 = __uint_as_float(0xbecccccdu);
f32 r10 = __uint_as_float(0xc1200000u);
f32 r11 = __uint_as_float(0x0u);
f32 r12 = __uint_as_float(0x3f800000u);
f32 r13 = __uint_as_float(0x7f7fffffu);
bool r14 = true;
bool r16_out_0 = false;
f32 r16_out_1 = __uint_as_float(0x7f800000u);
f32 r16_out_2 = 0.0f;
f32 r16_out_3 = 0.0f;
u32 r16_out_4 = 0u;
u32 r16_out_5 = 0u;
u32 r16_out_6 = 0u;
u32 r16_out_7 = 0u;
{
    hiprtRay _r;
    _r.origin    = { r8, r9, r10 };
    _r.direction = { r11, r11, r12 };
    _r.minT      = r11;
    _r.maxT      = r13;
    hiprtSceneTraversalClosest _tr((hiprtScene) r4, _r);
    hiprtHit _h = _tr.getNextHit();
    if (_h.hasHit()) {
        r16_out_0 = true;
        r16_out_1 = _h.t;
        r16_out_2 = _h.uv.x;
        r16_out_3 = _h.uv.y;
        r16_out_4 = _h.instanceID;
        r16_out_5 = _h.primID;
        r16_out_6 = ((const u32 *) r5)[_h.instanceID];
        r16_out_7 = ((const u32 *) r6)[_h.instanceID];
    }
}
bool r17 = r16_out_0;
((bool *) params.args[4])[r0] = r17;
u32 r18 = r16_out_7;
u32 r19 = 0x0u;
u32 r20 = r17 ? r18 : r19;
((u32 *) params.args[5])[r0] = r20;
bool r21 = r20 == r19;
u32 r22 = r17 ? r7 : (u32) 0;
((u32 *) params.args[6])[r0] = r22;
u32 r23 = r21 ? r22 : r20;
u32 r24 = r17 ? r23 : (u32) 0;
((u32 *) params.args[7])[r0] = r24;
f32 r25 = r16_out_1;
((f32 *) params.args[8])[r0] = r25;
f32 r26 = r16_out_2;
((f32 *) params.args[9])[r0] = r26;
f32 r27 = r16_out_3;
((f32 *) params.args[10])[r0] = r27;
u32 r28 = r16_out_5;
((u32 *) params.args[11])[r0] = r28;
}


