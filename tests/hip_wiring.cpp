/*
    tests/hip_wiring.cpp -- Phase 0a invariants for the HIP backend.

    Locks in the properties established by the Phase 0a wiring commit
    (PLAN.md §5). These were originally checked with a throwaway program; that
    verified nothing durable, so the checks live here instead where `ctest`
    runs them on every build.

    This file deliberately tests only what Phase 0a claims: that the backend is
    REGISTERED but INERT, and that adding it perturbs nothing else. Codegen
    behaviour is driven from tests/hip_codegen.cpp, which is expected to fail
    until Phase 2 lands.
*/

#include <drjit-core/jit.h>
#include <cstdio>
#include <cstring>

static int failures = 0;

static void check(bool cond, const char *what) {
    printf("  %-52s %s\n", what, cond ? "ok" : "FAIL");
    if (!cond)
        failures++;
}

int main(int, char **) {
    printf("hip_wiring: Phase 0a invariants\n");

    // --- Enum surface -------------------------------------------------------
    // Value 4 is not arbitrary: it must sit after Metal so that existing
    // serialized backend ids keep their meaning, and Count must follow it so
    // the per-backend arrays in State/malloc.cpp size themselves correctly.
    check((uint32_t) JitBackend::HIP == 4, "JitBackend::HIP == 4");
    check((uint32_t) JitBackend::Count == 5, "JitBackend::Count == 5");
    check(strcmp(jit_backend_name_v<JitBackend::HIP>, "hip") == 0,
          "jit_backend_name_v<HIP> == \"hip\"");

    // The name matters beyond cosmetics: it identifies registry domains, so a
    // collision or a stale "none" would silently misroute registered objects.
    check(strcmp(jit_backend_name_v<JitBackend::HIP>,
                 jit_backend_name_v<JitBackend::Metal>) != 0,
          "HIP and Metal backend names differ");

    // --- Registration is inert ---------------------------------------------
    jit_init((uint32_t) (1u << (uint32_t) JitBackend::LLVM) |
             (1u << (uint32_t) JitBackend::HIP));

    // Phase 0a contract: jitc_hip_init() returns false, so the backend never
    // enters state.backends and nothing downstream can dispatch to it. When
    // Phase 1 lands on real hardware this flips to 1 and the assertion below
    // must be updated deliberately, not silently.
    check(jit_has_backend(JitBackend::HIP) == 0,
          "jit_has_backend(HIP) == 0 (inert)");

    // --- Adding a backend perturbs nothing else -----------------------------
    check(jit_has_backend(JitBackend::LLVM) == 1,
          "jit_has_backend(LLVM) still 1");

    jit_shutdown(0);

    if (failures) {
        printf("hip_wiring: %d check(s) FAILED\n", failures);
        return 1;
    }
    printf("hip_wiring: all checks passed.\n");
    return 0;
}
