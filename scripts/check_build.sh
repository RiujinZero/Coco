#!/usr/bin/env bash
cd "$(dirname "$0")/.."
for a in laminar_train laminar_bytes laminar_chat; do g++ -std=c++20 -O1 -march=native -DNDEBUG -Icpp/include cpp/apps/$a.cpp cpp/src/organism.cpp -o /tmp/$a.chk 2>&1 | grep -E "error" | head -3; echo "$a exit ${PIPESTATUS[0]}"; done
