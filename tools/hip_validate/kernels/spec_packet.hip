// SPECIMEN: packet gather/scatter -- moving `count` contiguous elements per
// lane in one go. Machine-shaped SSA output. Self-checking; run with
// --expect-zero.
//
// Covers VarKind: PacketGather PacketScatter
//
// The whole opcode is an addressing contract, and the two halves scale their
// index DIFFERENTLY. Getting that backwards does not fail to compile; it reads
// and writes the wrong packet, which surfaces far away as a shuffled array.
//
//   PacketGather   byte address = ptr + index * (count * tsize)
//   PacketScatter  byte address = ptr + index * tsize
//
// They differ because op.cpp has already multiplied the SCATTER index by
// `count` before the opcode sees it, and not the gather index. Both the CUDA
// and Metal backends do it this way (cuda_packet.cpp's two `mad.wide` forms,
// metal_packet.cpp's two base-pointer expressions), so it is a contract of the
// IR rather than a backend convention -- which is why it is pinned here.
//
// The other reason this file exists: HIP does NOT need the wide-vector
// machinery the other two backends carry. Metal reinterprets through
// `uint4`/`as_type<>` and CUDA through `ld.global.v4`, because MSL and PTX will
// not merge adjacent accesses for them. HIP compiles C++ through LLVM, which
// does. Measured on gfx90a: the element-wise form below and an explicit
// `float4` form produce byte-identical machine code -- one global_load_dwordx4
// and one global_store_dwordx4 -- even when the pointer arrives as an opaque
// void* from the params struct, because the `index * 16` stride is enough for
// LLVM to infer the alignment. So the emitter stays element-wise. That also
// makes it correct for widths a chunked form would have to step down for --
// jit_var_gather_packet() requires a power of two today, so this is robustness
// rather than a requirement, but it costs nothing.

DRJIT_KERNEL void k(float *out, const float *in, unsigned n) {
    uint32_t r0 = DRJIT_TID;
    if (r0 >= n) return;

    uint32_t err = 0u;

    // A small array standing in for a device buffer of 4-wide packets. Values
    // are chosen so a swapped element or a mis-scaled index is visible.
    float buf[16];
    for (int i = 0; i < 16; ++i)
        buf[i] = (float) (100 + i);

    const uint32_t count = 4u;
    const uint32_t tsize = 4u;                    // sizeof(float)
    const uint32_t total_bytes = count * tsize;   // 16

    // --- PacketGather: index is in PACKETS ----------------------------------
    {
        uint32_t index = r0 % 4u;                 // packet 0..3

        const float *base =
            (const float *) ((const char *) buf + (size_t) index * total_bytes);

        float r1 = base[0], r2 = base[1], r3 = base[2], r4 = base[3];

        // Packet `index` holds elements [index*4, index*4+4)
        float want = (float) (100 + index * 4);
        if (r1 != want)        err |= 1u << 0;
        if (r2 != want + 1.0f) err |= 1u << 1;
        if (r3 != want + 2.0f) err |= 1u << 2;
        if (r4 != want + 3.0f) err |= 1u << 3;

        // Scaling the gather index by tsize instead of total_bytes is the
        // classic error: it still lands inside the buffer, so nothing traps.
        const float *wrong =
            (const float *) ((const char *) buf + (size_t) index * tsize);
        if (index != 0u && wrong[0] == r1) err |= 1u << 4;
    }

    // --- PacketGather, masked: off lanes read zero, not garbage --------------
    //
    // The masked form must still DEFINE every output: they are referenced
    // unconditionally further down the kernel, so a lane that skipped the load
    // would propagate an uninitialized register.
    {
        bool mask = (r0 % 2u) == 0u;
        uint32_t index = 1u;

        const float *base =
            (const float *) ((const char *) buf + (size_t) index * total_bytes);

        float r5 = mask ? base[0] : 0.0f;
        float r6 = mask ? base[1] : 0.0f;

        if (mask) {
            if (r5 != 104.0f || r6 != 105.0f) err |= 1u << 5;
        } else {
            if (r5 != 0.0f || r6 != 0.0f) err |= 1u << 6;
        }
    }

    // --- PacketScatter: index is ALREADY scaled by count --------------------
    {
        float dst[16];
        for (int i = 0; i < 16; ++i)
            dst[i] = -1.0f;

        uint32_t packet = r0 % 4u;
        uint32_t index  = packet * count;         // what op.cpp hands the opcode

        float *base = (float *) ((char *) dst + (size_t) index * tsize);
        base[0] = 10.0f; base[1] = 11.0f; base[2] = 12.0f; base[3] = 13.0f;

        uint32_t at = packet * count;
        if (dst[at + 0] != 10.0f) err |= 1u << 7;
        if (dst[at + 1] != 11.0f) err |= 1u << 8;
        if (dst[at + 2] != 12.0f) err |= 1u << 9;
        if (dst[at + 3] != 13.0f) err |= 1u << 10;

        // Everything outside the packet must be untouched -- a scatter that
        // re-scaled the index by total_bytes would write four packets apart.
        uint32_t other = (at + 4u) % 16u;
        if (dst[other] != -1.0f) err |= 1u << 11;
    }

    // --- A packet width that is not a power of two ---------------------------
    //
    // count == 3 has no naturally aligned wide access. Element-wise emission
    // stays correct here for free; a chunked form has to notice and step down.
    {
        float dst3[12];
        for (int i = 0; i < 12; ++i)
            dst3[i] = -1.0f;

        uint32_t packet = r0 % 4u;
        uint32_t index  = packet * 3u;

        float *base = (float *) ((char *) dst3 + (size_t) index * tsize);
        base[0] = 20.0f; base[1] = 21.0f; base[2] = 22.0f;

        if (dst3[packet * 3u + 0] != 20.0f) err |= 1u << 12;
        if (dst3[packet * 3u + 1] != 21.0f) err |= 1u << 13;
        if (dst3[packet * 3u + 2] != 22.0f) err |= 1u << 14;
    }

    out[r0] = (float) err;
}
