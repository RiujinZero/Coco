#!/usr/bin/env bash
# Build the native pipeline under WSL/Linux with GCC (primary bench target: -O3 -march=native -flto).
#   scripts/build_wsl.sh            # Release build into build-wsl/
#   scripts/build_wsl.sh pgo-gen    # instrumented build
#   scripts/build_wsl.sh pgo-use    # optimised with collected profiles
# Uses CMake when available, otherwise invokes g++ directly with the same flags.
set -euo pipefail
cd "$(dirname "$0")/../cpp"
mode="${1:-release}"
out=../build-wsl
mkdir -p "$out"
if command -v cmake >/dev/null 2>&1; then
  extra=""
  case "$mode" in
    pgo-gen) extra="-DLAMINAR_PGO_GEN=ON" ;;
    pgo-use) extra="-DLAMINAR_PGO_USE=ON" ;;
  esac
  cmake -S . -B "$out" -DCMAKE_BUILD_TYPE=Release $extra >/dev/null
  cmake --build "$out" -j"$(nproc)"
else
  flags="-std=c++20 -O3 -march=native -flto -DNDEBUG -Wall -Wextra -Wno-unused-parameter -Iinclude"
  case "$mode" in
    pgo-gen) flags="$flags -fprofile-generate" ;;
    pgo-use) flags="$flags -fprofile-use -fprofile-correction" ;;
  esac
  build_pids=()
  for t in laminar_synth laminar_bytes laminar_bench laminar_needle laminar_generate laminar_chat laminar_train; do
    g++ $flags apps/$t.cpp src/organism.cpp -o "$out/$t" &
    build_pids+=("$!")
  done
  g++ $flags tests/test_kernels.cpp src/organism.cpp -o "$out/test_kernels" &
  build_pids+=("$!")
  g++ $flags tests/test_packed_allocation.cpp src/organism.cpp -o "$out/test_packed_allocation" &
  build_pids+=("$!")
  g++ $flags tests/test_grounded_teaching.cpp -o "$out/test_grounded_teaching" &
  build_pids+=("$!")
  g++ $flags tests/test_grounded_archive.cpp -o "$out/test_grounded_archive" &
  build_pids+=("$!")
  g++ $flags tests/test_backprop_choice.cpp -o "$out/test_backprop_choice" &
  build_pids+=("$!")
  g++ $flags tests/test_policy_discovery.cpp -o "$out/test_policy_discovery" &
  build_pids+=("$!")
  g++ $flags tests/test_concept_memory.cpp -o "$out/test_concept_memory" &
  build_pids+=("$!")
  g++ $flags tests/test_relational_session.cpp -o "$out/test_relational_session" &
  build_pids+=("$!")
  g++ $flags tests/test_relational_control.cpp -o "$out/test_relational_control" &
  build_pids+=("$!")
  for t in computational_session composed_discovery composed_learner composed_session composed_support capacity continual_guidance goal_suspension episode_return concept_isolation; do
    g++ $flags "tests/test_$t.cpp" -o "$out/test_$t" &
    build_pids+=("$!")
  done
  build_failed=0
  for build_pid in "${build_pids[@]}"; do
    if ! wait "$build_pid"; then build_failed=1; fi
  done
  # A bare `wait` masks failed background builds and can run stale executables.
  if ((build_failed)); then exit 1; fi
fi
"$out/test_kernels"
"$out/test_packed_allocation"
"$out/test_grounded_teaching"
"$out/test_grounded_archive"
"$out/test_backprop_choice"
"$out/test_policy_discovery"
"$out/test_concept_memory"
"$out/test_relational_session"
"$out/test_relational_control"
"$out/test_computational_session"
"$out/test_composed_discovery"
"$out/test_composed_learner"
"$out/test_composed_support"
"$out/test_composed_session"
"$out/test_capacity"
"$out/test_continual_guidance"
"$out/test_goal_suspension"
"$out/test_episode_return"
"$out/test_concept_isolation"
