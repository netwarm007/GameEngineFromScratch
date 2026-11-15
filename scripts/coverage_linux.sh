#!/bin/bash
# Build with Clang coverage instrumentation, run CTest, and emit llvm-cov reports.
# Usage: scripts/coverage_linux.sh [build-dir] [report-dir]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${1:-$ROOT/build-coverage}"
REPORT_DIR="${2:-$ROOT/coverage}"

find_tool() {
    local name="$1" candidate
    if command -v "$name" >/dev/null 2>&1; then command -v "$name"; return; fi
    for candidate in $(compgen -c "$name-" | sort -rV | uniq); do
        if [[ "$candidate" =~ ^$name-[0-9]+$ ]]; then command -v "$candidate"; return; fi
    done
    echo "$name not found" >&2; exit 1
}
LLVM_COV="$(find_tool llvm-cov)"
LLVM_PROFDATA="$(find_tool llvm-profdata)"

export CC="${CC:-clang}" CXX="${CXX:-clang++}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$ROOT/External/$(uname -s)/lib"

cmake -S "$ROOT" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON
cmake --build "$BUILD_DIR"

# Tests run from their own working directories, so the profile path must be absolute.
REPORT_DIR="$(mkdir -p "$REPORT_DIR" && cd "$REPORT_DIR" && pwd)"
rm -rf "$REPORT_DIR" && mkdir -p "$REPORT_DIR/profraw"
# %p/%m keep profiles of concurrently running test processes apart.
export LLVM_PROFILE_FILE="$REPORT_DIR/profraw/%p-%m.profraw"
ctest --test-dir "$BUILD_DIR" --output-on-failure || echo "ctest reported failures; coverage still generated"

"$LLVM_PROFDATA" merge -sparse "$REPORT_DIR"/profraw/*.profraw -o "$REPORT_DIR/coverage.profdata"

# Every instrumented executable under the build dir contributes to the report.
mapfile -t EXES < <(find "$BUILD_DIR" -type f -executable ! -name '*.so*' ! -name '*.sh' ! -path '*/CMakeFiles/*')
OBJECTS=()
for exe in "${EXES[@]:1}"; do OBJECTS+=("-object" "$exe"); done
FIRST="${EXES[0]}"

IGNORE=(-ignore-filename-regex='(External|/Test/|/usr/|CMakeFiles|MGEMX\.(tab|yy|lex))')
COMMON=("$FIRST" "${OBJECTS[@]}" -instr-profile="$REPORT_DIR/coverage.profdata" "${IGNORE[@]}")

"$LLVM_COV" export "${COMMON[@]}" -format=lcov > "$REPORT_DIR/lcov.info"
"$LLVM_COV" show "${COMMON[@]}" -format=html -output-dir="$REPORT_DIR/html"
"$LLVM_COV" report "${COMMON[@]}" | tee "$REPORT_DIR/summary.txt"
