/*
    src/hip_format.h -- reindent generated HIP source for human consumption

    Code generation emits UNFORMATTED source: tracking indentation while
    emitting is costly and pointless for the compiler, which does not care.
    This pass produces a readable copy on demand, for JitFlag.PrintIR and high
    log levels.

    Ported from jitc_metal_format() (src/metal_eval.cpp), but factored as a
    PURE std::string -> std::string function rather than operating on the
    global code buffer. That is deliberate: a pure function is directly
    testable (tests/hip_format.cpp) with no JIT state, no device, and no
    exported symbols, whereas the Metal original can only be exercised through
    a full assembly. jitc_hip_format() in hip_eval.cpp wraps this with the
    buffer swap.

    The algorithm assumes three properties of the emitted text:

      1. every { and } relates to control flow (no braces in string literals
         or initializer lists),
      2. the only comments are // line comments,
      3. no statement is split across a blank or comment line.

    Emitted HIP satisfies all three. Violating them degrades indentation
    quality but must NEVER change content -- tests/hip_format.cpp asserts that
    invariant directly, including on input that breaks assumption 1.
*/

#pragma once

#include <string>
#include <cstddef>

/// Reindent `n` bytes of generated source. Existing leading and trailing
/// horizontal whitespace on each line is discarded, so the pass is idempotent.
inline std::string jitc_hip_reindent(const char *src, size_t n) {
    std::string out;
    out.reserve(n + n / 4);

    int  depth = 0;
    bool stmt_open = false; // previous code line left a statement unterminated

    for (size_t i = 0; i < n; ) {
        // Carve out one line and trim its horizontal whitespace.
        size_t b = i;
        while (i < n && src[i] != '\n')
            i++;
        size_t e = i;
        if (i < n)
            i++; // consume '\n'
        while (b < e && (src[b] == ' ' || src[b] == '\t'))
            b++;
        while (e > b && (src[e - 1] == ' ' || src[e - 1] == '\t'))
            e--;

        if (b == e) { // blank line: also closes any open statement
            out += '\n';
            stmt_open = false;
            continue;
        }

        bool is_comment = src[b] == '/' && b + 1 < e && src[b + 1] == '/';
        bool is_preproc = src[b] == '#';

        // Scan for brace deltas and the last significant character, stopping at
        // a trailing // comment so its braces do not count. Scene properties
        // are emitted as comments (BACKEND_NOTES §8), so this matters.
        int  opens = 0, closes = 0, leading_close = 0;
        bool seen = false;
        char last = 0;
        for (size_t k = b; k < e; k++) {
            char c = src[k];
            if (c == '/' && k + 1 < e && src[k + 1] == '/')
                break;
            if (c == '{') {
                opens++;
                seen = true;
            } else if (c == '}') {
                closes++;
                if (!seen)
                    leading_close++;
            } else if (c != ' ' && c != '\t') {
                seen = true;
            }
            if (c != ' ' && c != '\t')
                last = c;
        }

        // A line that *starts* with } dedents before printing, not after.
        int indent = depth - leading_close;
        if (indent < 0)
            indent = 0;
        if (stmt_open && !is_preproc && !is_comment)
            indent++; // hanging indent for continuation lines

        // Preprocessor directives go to column 0 regardless of nesting. This
        // differs from the Metal original, which indents them to `depth`.
        // Column 0 is the universal C convention and keeps directives
        // scannable in a multi-thousand-line dump (§0.2 measured 2039 lines
        // for the simplest scene).
        if (is_preproc)
            indent = 0;

        out.append((size_t) indent * 4, ' ');
        out.append(src + b, e - b);
        out += '\n';

        depth += opens - closes;
        if (depth < 0)
            depth = 0;

        stmt_open = !(is_comment || is_preproc || last == ';' || last == '{' ||
                      last == '}');
    }

    return out;
}
