/*
    tests/hip_format.cpp -- the generated-source reindent pass.

    Written before src/hip_format.h exists.

    Codegen emits UNFORMATTED source because tracking indentation during
    emission is costly; this pass reindents on demand for JitFlag.PrintIR and
    high log levels. It is not cosmetic: §0.2 measured a 2039-line kernel for
    the *simplest* Mitsuba scene, and readable dumps are a large part of the
    §3.1 debuggability argument that justified emitting source at all.

    Metal's equivalent (jitc_metal_format) relies on three properties of the
    emitted text, which emitted HIP also satisfies:

      1. braces only ever denote control flow,
      2. the only comments are // line comments,
      3. no statement is split across a blank or comment line.

    Violating them degrades indentation but must never change semantics -- so
    the last case below asserts the pass is content-preserving no matter what.
*/

#include "../src/hip_format.h"
#include <cstdio>
#include <cstring>
#include <string>

static int failures = 0;

static void expect(const char *name, const std::string &in,
                   const std::string &want) {
    std::string got = jitc_hip_reindent(in.c_str(), in.size());
    if (got == want) {
        printf("  %-46s ok\n", name);
    } else {
        printf("  %-46s FAIL\n", name);
        printf("    --- got ---\n%s    --- want ---\n%s", got.c_str(), want.c_str());
        failures++;
    }
}

static void check(bool cond, const char *what) {
    printf("  %-46s %s\n", what, cond ? "ok" : "FAIL");
    if (!cond)
        failures++;
}

int main(int, char **) {
    printf("hip_format: generated-source reindent\n");

    // Nesting: four spaces per level.
    expect("basic nesting",
           "void k() {\n"
           "int a = 1;\n"
           "if (a) {\n"
           "a = 2;\n"
           "}\n"
           "}\n",
           "void k() {\n"
           "    int a = 1;\n"
           "    if (a) {\n"
           "        a = 2;\n"
           "    }\n"
           "}\n");

    // A line that *starts* with a close brace dedents before printing, not
    // after -- otherwise every closing brace sits one level too deep.
    expect("leading close brace dedents itself",
           "if (x) {\n"
           "y();\n"
           "}\n",
           "if (x) {\n"
           "    y();\n"
           "}\n");

    // Continuation lines get a hanging indent. The previous line not ending in
    // ; { or } is what marks a statement as still open.
    expect("hanging indent for continuations",
           "void k() {\n"
           "int a = foo(1,\n"
           "2);\n"
           "int b = 3;\n"
           "}\n",
           "void k() {\n"
           "    int a = foo(1,\n"
           "        2);\n"
           "    int b = 3;\n"
           "}\n");

    // Braces inside a trailing comment must not move the depth. Scene
    // configuration is emitted as comments (BACKEND_NOTES §8), so this is a
    // real case, not a hypothetical.
    expect("braces in // comments are ignored",
           "void k() {\n"
           "int a = 1; // } not a real brace {\n"
           "int b = 2;\n"
           "}\n",
           "void k() {\n"
           "    int a = 1; // } not a real brace {\n"
           "    int b = 2;\n"
           "}\n");

    // Preprocessor directives sit at column 0 regardless of nesting depth --
    // the universal C convention, and it makes them scannable in a 2000-line
    // dump.
    //
    // NOTE: deliberately NOT tested here is what a directive does to an *open
    // statement* (e.g. a #define between the two halves of a split
    // expression). Assumption 3 above says that cannot occur, so any behaviour
    // there is undefined and asserting it would pin down an accident.
    expect("preprocessor directives sit at column 0",
           "void k() {\n"
           "#define X 1\n"
           "int a = X;\n"
           "}\n",
           "void k() {\n"
           "#define X 1\n"
           "    int a = X;\n"
           "}\n");

    // Existing leading whitespace is discarded, so the pass is idempotent --
    // running it twice must not drift.
    {
        std::string in = "void k() {\nint a = 1;\n}\n";
        std::string once  = jitc_hip_reindent(in.c_str(), in.size());
        std::string twice = jitc_hip_reindent(once.c_str(), once.size());
        check(once == twice, "idempotent (reindent twice == once)");
    }

    // Semantics must survive even when the three assumptions are violated.
    // Strip all whitespace from input and output; they must match exactly.
    {
        const char *nasty =
            "void k() {\n"
            "const char *s = \"a { brace in a string\";\n"
            "int a = 1;\n"
            "}\n";
        std::string out = jitc_hip_reindent(nasty, strlen(nasty));
        auto squeeze = [](const std::string &s) {
            std::string r;
            for (char c : s)
                if (c != ' ' && c != '\t' && c != '\n')
                    r += c;
            return r;
        };
        check(squeeze(out) == squeeze(nasty),
              "content-preserving even on brace-in-string");
    }

    // Degenerate inputs must not crash or invent content.
    expect("empty input", "", "");
    expect("trailing newline preserved", "a;\n", "a;\n");

    if (failures) {
        printf("hip_format: %d check(s) FAILED\n", failures);
        return 1;
    }
    printf("hip_format: all checks passed.\n");
    return 0;
}
