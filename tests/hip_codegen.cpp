/*
    tests/hip_codegen.cpp -- acceptance test driving Phase 2 (HIP codegen).

    ############################################################################
    #  PHASE 2 MILESTONE: PASSING.                                            #
    #                                                                          #
    #  Written before any codegen existed and registered WILL_FAIL, so ctest   #
    #  stayed green while the backend was a stub and would turn RED the moment #
    #  it started working. That tripwire has now fired and the property has    #
    #  been removed; this is a live regression guard rather than a prediction. #
    #                                                                          #
    #  Do not "fix" this test by weakening it. It encodes the Phase 2          #
    #  milestone from PLAN.md §5: `c = (a + b) * 5` correct on device.         #
    ############################################################################

    Written before the implementation, on purpose: it defines what "Phase 2
    works" means in executable terms rather than prose. Every opcode brought up
    in hip_eval.cpp should be accompanied by an assertion here.

    Note this is the END-TO-END acceptance gate and needs a real device. The
    fast inner loop for codegen -- which needs no AMD hardware at all -- is
    tools/hip_validate, which checks emitted source two ways (compiled for real
    gfx90a, and executed on NVIDIA). Use that while iterating; use this to say
    the phase is done.
*/

#include <drjit-core/jit.h>
#include <cstdio>
#include <cmath>

static int failures = 0;

static void check(bool cond, const char *what) {
    printf("  %-52s %s\n", what, cond ? "ok" : "FAIL");
    if (!cond)
        failures++;
}

int main(int, char **) {
    printf("hip_codegen: Phase 2 acceptance\n");

    jit_init(1u << (uint32_t) JitBackend::HIP);

    if (!jit_has_backend(JitBackend::HIP)) {
        // SKIP, not fail. In a default build the backend is deliberately inert
        // (Phase 0a), so there is nothing to exercise -- reporting failure there
        // would make an untouched configuration look broken. 77 is ctest's
        // conventional skip code; see SKIP_RETURN_CODE in tests/CMakeLists.txt.
        printf("  HIP backend unavailable -- build with -DDRJIT_HIP_CUDA_SHIM=ON\n"
               "  (or on real hardware) to exercise the Phase 2 milestone.\n");
        printf("hip_codegen: SKIPPED\n");
        return 77;
    }

    // ---- Phase 2 milestone: c = (a + b) * 5 -------------------------------
    constexpr uint32_t N = 1024;

    uint32_t a    = jit_var_counter(JitBackend::HIP, N),   // 0,1,2,...
             b    = jit_var_u32(JitBackend::HIP, 7),
             five = jit_var_u32(JitBackend::HIP, 5);

    uint32_t add_deps[2] = { a, b };
    uint32_t sum = jit_var_op(JitOp::Add, add_deps);

    uint32_t mul_deps[2] = { sum, five };
    uint32_t c = jit_var_op(JitOp::Mul, mul_deps);

    jit_var_eval(c);

    void *devptr = nullptr;
    uint32_t c_evaluated = jit_var_data(c, &devptr);

    uint32_t host[N];
    jit_memcpy(JitBackend::HIP, host, devptr, sizeof(host));

    bool all_ok = true;
    for (uint32_t i = 0; i < N; ++i)
        if (host[i] != (i + 7u) * 5u)
            all_ok = false;
    check(all_ok, "c = (a + b) * 5 correct for all lanes");

    jit_var_dec_ref(a);    jit_var_dec_ref(b);   jit_var_dec_ref(sum);
    jit_var_dec_ref(five); jit_var_dec_ref(c);   jit_var_dec_ref(c_evaluated);

    // ---- Beyond the milestone: the wider opcode surface --------------------
    //
    // The milestone above only exercises Add and Mul. These cover the opcodes
    // added alongside it, each of which has a matching specimen in
    // tools/hip_validate/kernels/. Values are chosen so a wrong intrinsic
    // shifts the result rather than happening to agree.
    {
        uint32_t idx  = jit_var_counter(JitBackend::HIP, N);
        uint32_t idxf = jit_var_cast(idx, VarType::Float32, 0);   // Cast

        uint32_t four = jit_var_f32(JitBackend::HIP, 4.0f);
        uint32_t d2[2] = { idxf, four };
        uint32_t scaled = jit_var_op(JitOp::Mul, d2);             // f32 Mul

        uint32_t sq_deps[1] = { scaled };
        uint32_t root = jit_var_op(JitOp::Sqrt, sq_deps);         // Sqrt

        // Min against a constant exercises the float spelling (fminf).
        uint32_t cap = jit_var_f32(JitBackend::HIP, 10.0f);
        uint32_t md[2] = { root, cap };
        uint32_t clamped = jit_var_op(JitOp::Min, md);            // Min

        // Floor, then compare -- exercises rounding and a bool-producing op.
        uint32_t fl_deps[1] = { clamped };
        uint32_t fl = jit_var_op(JitOp::Floor, fl_deps);          // Floor

        jit_var_eval(fl);
        void *fp = nullptr;
        uint32_t fl_eval = jit_var_data(fl, &fp);

        float hostf[N];
        jit_memcpy(JitBackend::HIP, hostf, fp, sizeof(hostf));

        bool ok = true;
        for (uint32_t i = 0; i < N; ++i) {
            float want = std::sqrt((float) i * 4.0f);
            if (want > 10.0f) want = 10.0f;
            want = std::floor(want);
            if (hostf[i] != want)
                ok = false;
        }
        check(ok, "Cast/Mul/Sqrt/Min/Floor chain correct for all lanes");

        jit_var_dec_ref(idx);  jit_var_dec_ref(idxf); jit_var_dec_ref(four);
        jit_var_dec_ref(scaled); jit_var_dec_ref(root); jit_var_dec_ref(cap);
        jit_var_dec_ref(clamped); jit_var_dec_ref(fl); jit_var_dec_ref(fl_eval);
    }

    // ---- Gather --------------------------------------------------------------
    //
    // A REVERSING gather (index = N-1-i) rather than an identity one: with
    // index == i, a backend that ignored the index entirely and returned the
    // lane's own element would still pass. Reversal makes that failure visible.
    {
        uint32_t src  = jit_var_counter(JitBackend::HIP, N);       // 0..N-1
        uint32_t nm1  = jit_var_u32(JitBackend::HIP, N - 1);
        uint32_t idx  = jit_var_counter(JitBackend::HIP, N);

        uint32_t sd[2] = { nm1, idx };
        uint32_t rev = jit_var_op(JitOp::Sub, sd);                 // N-1-i

        // Evaluate the source so the gather reads real memory rather than
        // folding into the producing expression.
        jit_var_eval(src);

        uint32_t g = jit_var_gather(src, rev, jit_var_bool(JitBackend::HIP, true));

        jit_var_eval(g);
        void *gp = nullptr;
        uint32_t g_eval = jit_var_data(g, &gp);

        uint32_t hostg[N];
        jit_memcpy(JitBackend::HIP, hostg, gp, sizeof(hostg));

        bool ok = true;
        for (uint32_t i = 0; i < N; ++i)
            if (hostg[i] != (N - 1 - i))
                ok = false;
        check(ok, "reversing gather correct for all lanes");

        jit_var_dec_ref(src); jit_var_dec_ref(nm1); jit_var_dec_ref(idx);
        jit_var_dec_ref(rev); jit_var_dec_ref(g);   jit_var_dec_ref(g_eval);
    }

    // ---- Control flow: NOT YET VERIFIED --------------------------------------
    //
    // jitc_hip_render() implements LoopStart/Cond/End/Phi/Output and
    // CondStart/Mid/End, ported closely from metal_eval.cpp, and it compiles.
    // It is NOT exercised here.
    //
    // A hand-rolled symbolic loop was attempted and removed: jit_var_loop_end()
    // requires a jit_record_begin() checkpoint and may return 0, meaning the
    // body must be recorded a second time after Dr.Jit simplifies the state.
    // Getting that protocol subtly wrong produces a test that fails for reasons
    // unrelated to codegen -- which is worse than no test, because it points at
    // the wrong suspect.
    //
    // The right coverage is to register HIP with the TEST_* macros in
    // tests/test.h so the existing test_loop / test_vcall suites run against
    // this backend, rather than reimplementing their protocol by hand. Until
    // then, treat control flow as WRITTEN BUT UNVERIFIED.

    jit_shutdown(0);

    if (failures) {
        printf("hip_codegen: %d check(s) FAILED\n", failures);
        return 1;
    }
    printf("hip_codegen: Phase 2 milestone verified on device.\n");
    return 0;
}
