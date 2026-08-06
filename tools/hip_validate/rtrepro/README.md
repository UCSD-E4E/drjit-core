# `rtrepro` — standalone reproducer for the §11n.1 HIP-RT builder failure

`rtbuild.cpp` calls `hiprtBuildTraceKernels()` on a source file with
**argument-for-argument identical** parameters to
`jitc_hip_shim_rt_compile()` in `src/hip_core.cpp`. It exists to answer one
question that kept the investigation honest:

> Is the HIP we emit invalid, or is HIP-RT's builder failing on valid input?

`failing_kernel.hip` is the exact source drjit-core logged on failure while
running Mitsuba's `test_mesh.py::test13 + test14` on `hip_ad_rgb`.

**The answer, so far: the source is valid.** It builds standalone, and the same
text fails inside the Mitsuba process. See BACKEND_NOTES §11n.1 for the full
elimination table — device memory, scene churn, kernel count, interleaving and a
missing CUDA context binding have all been ruled out by experiment.

## Build and run

From inside `nix develop --impure .#hip`:

```sh
g++ -std=c++17 rtbuild.cpp \
    -I$HIPRT_NV_PATH/include -I$DRJIT_HIP_SHIM_CUDA_INCLUDE \
    -L$HIPRT_NV_PATH/lib -L/run/opengl-driver/lib \
    -lhiprt64 -lcuda -Wl,-rpath,$HIPRT_NV_PATH/lib -o rtbuild

./rtbuild failing_kernel.hip        # entry point is read from the source
```

`-lcuda` needs `/run/opengl-driver/lib` on NixOS: the driver stub is not in the
CUDA package.

Exit status is 0 on a successful build, 1 on a HIP-RT failure.

## Capturing a fresh failing kernel

drjit-core dumps the source at `Warn` level when the build fails, so:

```py
import drjit as dr, pytest, sys
dr.set_log_level(dr.LogLevel.Warn)
sys.exit(pytest.main([...]))
```

then extract between the `generated source that HIP-RT failed to build:` marker
and the `Dr.Jit encountered an unrecoverable error` banner. Note that some
failures **segfault inside `hiprtBuildTraceKernels` itself**, in which case
nothing is logged — those need a debugger rather than this path.

## Next probes

1. Build the same kernel **twice against one `hiprtContext`**. The in-process
   case does this and the standalone case does not, which is currently the
   clearest untested difference.
2. Bisect `test13` to find which operation arms the failure — `test14` alone
   passes, so something in `test13` changes the state.
