/*
    hip_validate.cpp -- dual-path validation harness for emitted HIP kernels.

    Implements PLAN.md §0.3. Takes one emitted kernel and checks it two
    independent ways, neither of which needs AMD hardware:

        arm "gfx"   hipcc --offload-arch=gfx90a --genco
                    -> Is this VALID FOR THE REAL TARGET? Address spaces,
                       wave64 intrinsics, ISA validity. Compile only; the
                       resulting code object is inspected but not run.

        arm "exec"  NVRTC -> cuModuleLoadData -> cuLaunchKernel -> read back
                    -> Are the NUMBERS RIGHT? Executes on the local NVIDIA GPU.

    Together they cover the two ways codegen fails: invalid for the target, and
    valid but wrong. Neither alone is sufficient.

    The kernel source must contain NO platform includes -- this harness prepends
    prelude_hip.h or prelude_nvrtc.h depending on the arm. See
    prelude_portable.h for the contract that makes one source checkable twice.

    Kernel signature convention:

        DRJIT_KERNEL void k(float *out, const float *in, unsigned n)

    Both CUDA and HIP runtimes are resolved with dlopen rather than linked, so
    this tool builds and runs with either arm absent -- it degrades to running
    only the arm whose toolchain is present. That matters because the same tool
    should be useful on the MI210 host later, where there is no NVIDIA GPU.
*/

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>

// ---------------------------------------------------------------------------
//  NVRTC + CUDA driver ABI, declared rather than included (cf. drjit-core
//  src/cuda_api.h, which does the same thing for the CUDA backend).
// ---------------------------------------------------------------------------

typedef void *nvrtcProgram;
typedef int   nvrtcResult;
typedef int   CUresult;
typedef int   CUdevice;
typedef void *CUcontext, *CUmodule, *CUfunction;
typedef unsigned long long CUdeviceptr;

static nvrtcResult (*nvrtcCreateProgram)(nvrtcProgram *, const char *, const char *,
                                         int, const char **, const char **);
static nvrtcResult (*nvrtcCompileProgram)(nvrtcProgram, int, const char **);
static nvrtcResult (*nvrtcGetProgramLogSize)(nvrtcProgram, size_t *);
static nvrtcResult (*nvrtcGetProgramLog)(nvrtcProgram, char *);
static nvrtcResult (*nvrtcGetPTXSize)(nvrtcProgram, size_t *);
static nvrtcResult (*nvrtcGetPTX)(nvrtcProgram, char *);
static nvrtcResult (*nvrtcDestroyProgram)(nvrtcProgram *);
static nvrtcResult (*nvrtcVersion)(int *, int *);

static CUresult (*cuInit)(unsigned);
static CUresult (*cuDeviceGet)(CUdevice *, int);
static CUresult (*cuCtxCreate)(CUcontext *, unsigned, CUdevice);
static CUresult (*cuModuleLoadData)(CUmodule *, const void *);
static CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
static CUresult (*cuMemAlloc)(CUdeviceptr *, size_t);
static CUresult (*cuMemFree)(CUdeviceptr);
static CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void *, size_t);
static CUresult (*cuMemcpyDtoH)(void *, CUdeviceptr, size_t);
static CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                                  unsigned, unsigned, unsigned, unsigned,
                                  void *, void **, void **);
static CUresult (*cuCtxSynchronize)();

// ---------------------------------------------------------------------------

struct Options {
    std::string kernel_path;
    std::string entry     = "k";
    std::string arch;                 // defaults from $HIP_TARGET_ARCH
    std::string ref_path;
    unsigned    n         = 1024;
    double      tol       = 1e-5;
    bool        run_gfx   = true;
    bool        run_exec  = true;
    bool        dump_ptx  = false;
    bool        dump_isa  = false;
    bool        verbose   = false;
};

static std::string read_file(const std::string &p, bool *ok = nullptr) {
    std::ifstream f(p);
    if (!f) { if (ok) *ok = false; return {}; }
    std::stringstream ss; ss << f.rdbuf();
    if (ok) *ok = true;
    return ss.str();
}

static std::string env_or(const char *k, const char *fallback) {
    const char *v = getenv(k);
    return (v && *v) ? std::string(v) : std::string(fallback);
}

/// Deterministic input so results are reproducible across arms and runs.
static std::vector<float> make_input(unsigned n) {
    std::vector<float> v(n);
    for (unsigned i = 0; i < n; ++i)
        v[i] = std::sin((float) i * 0.017453292f) * 0.5f + 0.5f;
    return v;
}

static std::string prelude_dir() {
    // Preludes sit next to this source; the build stamps the path in.
    return env_or("HIP_VALIDATE_PRELUDE_DIR", DRJIT_HIP_VALIDATE_PRELUDE_DIR);
}

// ---------------------------------------------------------------------------
//  Arm "gfx": does it compile for the real target?
// ---------------------------------------------------------------------------

static bool run_gfx_arm(const Options &o, const std::string &kernel_src) {
    std::string hipcc = env_or("HIPCC", "hipcc");
    std::string devlib = env_or("HIP_DEVICE_LIB_PATH", "");

    bool ok = false;
    std::string prelude = read_file(prelude_dir() + "/prelude_hip.h", &ok);
    if (!ok) {
        printf("  [gfx ] SKIP  cannot read prelude_hip.h from %s\n",
               prelude_dir().c_str());
        return false;
    }

    std::string tmp_src = "/tmp/hip_validate_gfx.hip";
    std::string tmp_out = "/tmp/hip_validate_gfx.co";
    {
        std::ofstream f(tmp_src);
        // Inline the prelude rather than -include so the file is self-contained
        // and can be handed to a human verbatim when something goes wrong.
        f << prelude << "\n#line 1 \"" << o.kernel_path << "\"\n" << kernel_src;
    }

    std::string cmd = hipcc + " --offload-arch=" + o.arch + " --genco ";
    if (!devlib.empty())
        cmd += "--rocm-device-lib-path=" + devlib + " ";
    cmd += tmp_src + " -o " + tmp_out + " 2>&1";

    auto t0 = std::chrono::steady_clock::now();
    FILE *p = popen(cmd.c_str(), "r");
    if (!p) { printf("  [gfx ] FAIL  cannot invoke %s\n", hipcc.c_str()); return false; }
    std::string log; char buf[512];
    while (fgets(buf, sizeof(buf), p)) log += buf;
    int rc = pclose(p);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (rc != 0) {
        printf("  [gfx ] FAIL  hipcc exited %d (%.0f ms)\n", rc, ms);
        printf("%s", log.c_str());
        return false;
    }

    // A clean exit is already meaningful: hipcc rejects unknown targets with
    // "unsupported HIP gpu architecture", so we cannot silently get a code
    // object for the wrong arch.
    //
    // When clang-offload-bundler is available, confirm the bundle really
    // carries the requested target. Note this CANNOT be done by grepping the
    // code object: offload bundles are compressed, so the target string is not
    // present as a readable substring. The bundler is the only reliable check.
    std::string co = read_file(tmp_out);
    std::string bundler = env_or("HIP_BUNDLER", "");
    const char *tag_note = "  (target unconfirmed: set $HIP_BUNDLER to verify)";
    bool tagged = true;

    if (!bundler.empty()) {
        std::string lc = bundler + " --type=o --list --input=" + tmp_out + " 2>/dev/null";
        FILE *lp = popen(lc.c_str(), "r");
        std::string targets;
        if (lp) {
            while (fgets(buf, sizeof(buf), lp)) targets += buf;
            pclose(lp);
        }
        tagged = targets.find("amdgcn-amd-amdhsa--" + o.arch) != std::string::npos;
        tag_note = tagged ? "" : "  (bundle does NOT carry the requested target)";
    }

    printf("  [gfx ] %s  %s, %zu bytes, %.0f ms%s\n",
           tagged ? "PASS" : "FAIL",
           o.arch.c_str(), co.size(), ms, tag_note);

    if (o.dump_isa) {
        // Must unbundle first -- llvm-objdump cannot read an offload bundle.
        std::string objdump = env_or("HIP_OBJDUMP", "llvm-objdump");
        std::string elf = "/tmp/hip_validate_gfx.elf";
        if (!bundler.empty()) {
            std::string uc = bundler + " --type=o --unbundle --input=" + tmp_out +
                             " --output=" + elf + " --targets=hipv4-amdgcn-amd-amdhsa--" +
                             o.arch + " 2>/dev/null";
            if (system(uc.c_str()) == 0) {
                printf("  --- %s ISA ---\n", o.arch.c_str());
                std::string dis = objdump + " -d --mcpu=" + o.arch + " " + elf +
                                  " 2>/dev/null | sed -n '6,40p'";
                int ignored = system(dis.c_str()); (void) ignored;
            }
        } else {
            printf("  --dump-isa needs $HIP_BUNDLER (bundles must be unbundled first)\n");
        }
    }
    if (!o.verbose && !log.empty()) { /* warnings suppressed unless -v */ }
    else if (!log.empty()) printf("%s", log.c_str());

    return tagged;
}

// ---------------------------------------------------------------------------
//  Arm "exec": are the numbers right?
// ---------------------------------------------------------------------------

static bool load_nvidia() {
    void *rtc = dlopen("libnvrtc.so.12", RTLD_LAZY);
    if (!rtc) rtc = dlopen("libnvrtc.alt.so.12", RTLD_LAZY);
    if (!rtc) rtc = dlopen("libnvrtc.so", RTLD_LAZY);
    void *cu  = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!cu)  cu  = dlopen("libcuda.so", RTLD_LAZY);
    if (!rtc || !cu) return false;

#define S(h, f) *(void **) &f = dlsym(h, #f); if (!f) return false;
#define SV(h, f, name) *(void **) &f = dlsym(h, name); if (!f) return false;
    S(rtc, nvrtcCreateProgram) S(rtc, nvrtcCompileProgram)
    S(rtc, nvrtcGetProgramLogSize) S(rtc, nvrtcGetProgramLog)
    S(rtc, nvrtcGetPTXSize) S(rtc, nvrtcGetPTX)
    S(rtc, nvrtcDestroyProgram) S(rtc, nvrtcVersion)

    S(cu, cuInit) S(cu, cuDeviceGet) S(cu, cuModuleLoadData)
    S(cu, cuModuleGetFunction) S(cu, cuLaunchKernel) S(cu, cuCtxSynchronize)
    // Versioned entry points: the unversioned symbols resolve but are legacy
    // variants with different signatures, and fail at runtime with a
    // misleading CUDA_ERROR_INVALID_CONTEXT. drjit-core hits the same thing --
    // see LOAD(cuMemAlloc, "v2") in src/cuda_api.cpp.
    SV(cu, cuCtxCreate,  "cuCtxCreate_v2")
    SV(cu, cuMemAlloc,   "cuMemAlloc_v2")
    SV(cu, cuMemFree,    "cuMemFree_v2")
    SV(cu, cuMemcpyHtoD, "cuMemcpyHtoD_v2")
    SV(cu, cuMemcpyDtoH, "cuMemcpyDtoH_v2")
#undef S
#undef SV
    return true;
}

static bool run_exec_arm(const Options &o, const std::string &kernel_src,
                         std::vector<float> &out) {
    if (!load_nvidia()) {
        printf("  [exec] SKIP  libnvrtc/libcuda not loadable\n");
        return false;
    }

    bool ok = false;
    std::string prelude = read_file(prelude_dir() + "/prelude_nvrtc.h", &ok);
    if (!ok) {
        printf("  [exec] SKIP  cannot read prelude_nvrtc.h\n");
        return false;
    }
    std::string src = prelude + "\n#line 1 \"" + o.kernel_path + "\"\n" + kernel_src;

    auto t0 = std::chrono::steady_clock::now();
    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, src.c_str(), "k.cu", 0, nullptr, nullptr)) {
        printf("  [exec] FAIL  nvrtcCreateProgram\n"); return false;
    }
    const char *opts[] = { "--gpu-architecture=compute_86", "--std=c++17" };
    nvrtcResult crc = nvrtcCompileProgram(prog, 2, opts);
    if (crc) {
        size_t ls = 0; nvrtcGetProgramLogSize(prog, &ls);
        std::vector<char> log(ls + 1, 0);
        nvrtcGetProgramLog(prog, log.data());
        printf("  [exec] FAIL  NVRTC compile\n%s\n", log.data());
        return false;
    }
    size_t sz = 0; nvrtcGetPTXSize(prog, &sz);
    std::vector<char> ptx(sz);
    nvrtcGetPTX(prog, ptx.data());
    nvrtcDestroyProgram(&prog);
    auto t1 = std::chrono::steady_clock::now();
    double compile_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (o.dump_ptx) printf("  --- PTX ---\n%s\n", ptx.data());

    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
    if (cuInit(0) || cuDeviceGet(&dev, 0) || cuCtxCreate(&ctx, 0, dev)) {
        printf("  [exec] FAIL  CUDA init\n"); return false;
    }
    if (cuModuleLoadData(&mod, ptx.data())) {
        printf("  [exec] FAIL  cuModuleLoadData\n"); return false;
    }
    if (cuModuleGetFunction(&fn, mod, o.entry.c_str())) {
        printf("  [exec] FAIL  no entry point '%s'\n", o.entry.c_str()); return false;
    }

    std::vector<float> in = make_input(o.n);
    CUdeviceptr d_out = 0, d_in = 0;
    size_t bytes = (size_t) o.n * sizeof(float);
    if (cuMemAlloc(&d_out, bytes) || cuMemAlloc(&d_in, bytes) ||
        cuMemcpyHtoD(d_in, in.data(), bytes)) {
        printf("  [exec] FAIL  device allocation\n"); return false;
    }

    unsigned n = o.n;
    void *args[] = { &d_out, &d_in, &n };
    unsigned block = 128, grid = (o.n + block - 1) / block;
    if (cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, nullptr, args, nullptr) ||
        cuCtxSynchronize()) {
        printf("  [exec] FAIL  launch\n"); return false;
    }

    out.resize(o.n);
    if (cuMemcpyDtoH(out.data(), d_out, bytes)) {
        printf("  [exec] FAIL  readback\n"); return false;
    }
    cuMemFree(d_out); cuMemFree(d_in);

    // A kernel that "runs" but emits NaN/Inf everywhere is a silent failure
    // mode worth catching here rather than downstream.
    unsigned bad = 0;
    for (float v : out) if (!std::isfinite(v)) ++bad;
    if (bad) {
        printf("  [exec] FAIL  %u/%u non-finite outputs\n", bad, o.n);
        return false;
    }

    printf("  [exec] PASS  %u elems, compile %.0f ms, out[0]=%g out[%u]=%g\n",
           o.n, compile_ms, out[0], o.n - 1, out[o.n - 1]);
    return true;
}

// ---------------------------------------------------------------------------

static bool compare_ref(const Options &o, const std::vector<float> &out) {
    bool ok = false;
    std::string txt = read_file(o.ref_path, &ok);
    if (!ok) { printf("  [ref ] FAIL  cannot read %s\n", o.ref_path.c_str()); return false; }

    std::vector<float> ref;
    std::stringstream ss(txt);
    float v;
    while (ss >> v) ref.push_back(v);

    if (ref.size() != out.size()) {
        printf("  [ref ] FAIL  size %zu != %zu\n", ref.size(), out.size());
        return false;
    }
    double worst = 0; size_t at = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = std::fabs((double) ref[i] - (double) out[i]);
        if (d > worst) { worst = d; at = i; }
    }
    bool pass = worst <= o.tol;
    printf("  [ref ] %s  max |diff| = %.3g at [%zu] (tol %.3g)\n",
           pass ? "PASS" : "FAIL", worst, at, o.tol);
    return pass;
}

static void usage() {
    printf(
    "hip_validate -- dual-path validation for emitted HIP kernels (PLAN.md §0.3)\n\n"
    "usage: hip_validate [options] <kernel.hip>\n\n"
    "  The kernel must use only the prelude_portable.h macro subset and carry\n"
    "  NO platform includes; the matching prelude is prepended per arm.\n\n"
    "  Signature: DRJIT_KERNEL void k(float *out, const float *in, unsigned n)\n\n"
    "options:\n"
    "  --entry NAME    entry point (default: k)\n"
    "  --arch ARCH     AMD target (default: $HIP_TARGET_ARCH, else gfx90a)\n"
    "  --n N           element count (default: 1024)\n"
    "  --ref FILE      compare execution output against whitespace-separated floats\n"
    "  --tol T         comparison tolerance (default: 1e-5)\n"
    "  --no-gfx        skip the gfx compile arm\n"
    "  --no-exec       skip the NVIDIA execution arm\n"
    "  --dump-ptx      print NVRTC PTX\n"
    "  --dump-isa      print gfx ISA disassembly\n"
    "  -v              show compiler warnings\n");
}

int main(int argc, char **argv) {
    Options o;
    o.arch = env_or("HIP_TARGET_ARCH", "gfx90a");

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char *what) -> std::string {
            if (i + 1 >= argc) { printf("missing value for %s\n", what); exit(2); }
            return argv[++i];
        };
        if      (a == "--entry")    o.entry    = next("--entry");
        else if (a == "--arch")     o.arch     = next("--arch");
        else if (a == "--ref")      o.ref_path = next("--ref");
        else if (a == "--n")        o.n        = (unsigned) atoi(next("--n").c_str());
        else if (a == "--tol")      o.tol      = atof(next("--tol").c_str());
        else if (a == "--no-gfx")   o.run_gfx  = false;
        else if (a == "--no-exec")  o.run_exec = false;
        else if (a == "--dump-ptx") o.dump_ptx = true;
        else if (a == "--dump-isa") o.dump_isa = true;
        else if (a == "-v")         o.verbose  = true;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (!a.empty() && a[0] == '-') { printf("unknown option %s\n", a.c_str()); return 2; }
        else o.kernel_path = a;
    }

    if (o.kernel_path.empty()) { usage(); return 2; }

    bool ok = false;
    std::string kernel_src = read_file(o.kernel_path, &ok);
    if (!ok) { printf("cannot read %s\n", o.kernel_path.c_str()); return 2; }

    printf("hip_validate: %s (entry '%s', arch %s)\n",
           o.kernel_path.c_str(), o.entry.c_str(), o.arch.c_str());

    bool gfx_ok = true, exec_ok = true, ref_ok = true;
    std::vector<float> out;

    if (o.run_gfx)  gfx_ok  = run_gfx_arm(o, kernel_src);
    if (o.run_exec) exec_ok = run_exec_arm(o, kernel_src, out);
    if (!o.ref_path.empty() && o.run_exec && exec_ok) ref_ok = compare_ref(o, out);

    bool all = gfx_ok && exec_ok && ref_ok;
    printf("  ==> %s\n", all ? "OK" : "FAILED");

    // Reminder rather than a failure: the execution arm is warp-32, so wave64
    // semantics remain unverified until MI210 access (PLAN.md §0.3, §7.2).
    if (all && o.run_exec &&
        kernel_src.find("DRJIT_SHFL")   != std::string::npos)
        printf("  note: uses wave ops; width-64 semantics NOT verified on NVIDIA (§7.2)\n");
    if (all && o.run_exec &&
        (kernel_src.find("DRJIT_BALLOT") != std::string::npos ||
         kernel_src.find("DRJIT_ACTIVEMASK") != std::string::npos))
        printf("  note: uses lane masks; 64-bit ballot NOT verified on NVIDIA (§7.2)\n");

    return all ? 0 : 1;
}
