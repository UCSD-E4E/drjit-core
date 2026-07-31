/*
    prelude_portable.h -- the portable subset contract for emitted HIP kernels.

    This file is DOCUMENTATION, not code: it defines the macro surface that a
    Dr.Jit HIP source emitter is allowed to use, and that both `prelude_hip.h`
    (AMD / hipcc) and `prelude_nvrtc.h` (NVIDIA / NVRTC) must implement.

    The rule that makes dual validation possible (PLAN.md §0.3):

        EMITTED KERNELS CONTAIN NO PLATFORM INCLUDES.

    The emitter produces a kernel body using only the macros below. The harness
    prepends whichever prelude matches the arm being run. One emitted source is
    therefore checkable two independent ways -- compiled for real gfx90a by
    hipcc, and compiled *and executed* on NVIDIA by NVRTC -- without the emitter
    knowing which platform it is destined for.

    This is the smallest concrete instance of the emit abstraction that §3.1
    requires for source<->IR reversibility and multi-vendor optionality. Keep
    divergence between platforms confined to the prelude files; if a construct
    cannot be expressed here, that is a signal it needs an abstraction, not an
    #ifdef at the emission site.

    -------------------------------------------------------------------------
    Contract
    -------------------------------------------------------------------------

    DRJIT_KERNEL              Kernel entry-point decoration.
    DRJIT_DEVICE              Device-only function decoration.

    DRJIT_WARP_SIZE           Compile-time wave/warp width. 64 on CDNA/GCN,
                              32 on RDNA and all NVIDIA parts. This is §3.3's
                              single-definition parameter -- never hardcode 64
                              at an emission site, read it from here. HIP-RT
                              itself uses exactly this pattern
                              (hiprt_common.h: constexpr uint32_t WarpSize).

    DRJIT_LANE_MASK_T         Integer type wide enough to hold one bit per lane.
                              uint64_t on both platforms deliberately: HIP's
                              ballot is already 64-bit, CUDA's is 32-bit and is
                              widened. Emitted code must not assume 32 bits.

    DRJIT_TID                 Global thread index (uint32_t).
    DRJIT_SHFL(v, lane)       Read `v` from `lane` within the wave.
    DRJIT_BALLOT(pred)        One bit per lane where `pred` holds.
    DRJIT_ACTIVEMASK()        Currently-active lanes.

    -------------------------------------------------------------------------
    Wave-width warning
    -------------------------------------------------------------------------

    On the NVIDIA arm DRJIT_WARP_SIZE is 32, so wave64 *semantics* are NOT
    verified there -- only structure. §0.2 measured 24 shfl plus vote and
    activemask in a Cornell box at 1 spp, so these are the hot path of every
    render. Anything whose correctness depends on the width being 64 must be
    tracked as unverified until MI210 access (PLAN.md §0.3, §7.2).
*/

#error "prelude_portable.h documents the contract; include a concrete prelude instead."
