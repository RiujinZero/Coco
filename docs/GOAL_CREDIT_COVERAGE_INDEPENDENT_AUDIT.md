# Independent audit of the 512-update continuation

The frozen record in `logs/goal_credit_coverage_continuation_20260923_v1` passes the independent auditor `python/audit_goal_credit_coverage.py`. The machine result and exact auditor sources are preserved in `logs/goal_credit_coverage_independent_audit_v1`. This verifies the protocol and reported failure; it does not establish autonomous acquisition.

The audit verifies 56 frozen source files, seven predeclared files, five original comparator inputs, all 560 missions and 46,318 decision events. It independently reconstructs 37,316 execution arrival bits, 11,038 world callbacks and 1,010 terminal labels. It reproduces all 12,233 unassisted zero-epsilon choices with a scalar float32 forward pass over exported checkpoint parameters. All costs are exactly zero and utility is recomputed from the actual terminal outcome and decision count.

All 100 intermediate warm weight snapshots and warm mission results match the original pilot exactly. The continuing factual and random-credit arms start from the same warm parameters; neither receives post-warm hints. Each has 200 autonomous missions with 512 terminal updates per mission. The reused frozen-policy comparison is accepted only after checking its original configuration and all 201 original weight snapshots against that warm state.

| Arm | Sampled autonomous success | Greedy before consolidation | Greedy after consolidation |
|---|---:|---:|---:|
| Factual credit, 512 updates | 5/200 | 0/12 | 0/12 |
| Random credit, 512 updates | 8/200 | 0/12 | 0/12 |
| Original frozen-policy comparator | 5/200 | 0/12 | Not run |

The factual endpoint uses 2,440 distinct credited observations and 97,600 additional consolidation fits; the random endpoint uses 386 observations and 98,816 fits. Both remain below the declared 100,000-fit cap. Consolidation occurs on separate clones. Before/after empirical exports are byte-identical, semantic versions and credit counts are unchanged, and the audit independently verifies every exported neural and empirical menu choice and its trace-derived support counts.

The separate stopping summary matches an independent route-based reconstruction for every mission. Before consolidation, the factual endpoint completes the physical world and eventually finishes it in all 12 cases, but performs 312 extra `act` decisions while its cached answer is already true and misses every urgent deadline. After consolidation it completes the world in all 12 cases but never finishes it, making 2,451 cached-true non-finish decisions. These are reason operations in the recorded traces. Random-credit endpoints also miss every urgent deadline. Physical completion alone is not scored as mission success.

The unchanged warm data contain 4,573 teacher-labelled decisions, 67 unique exact observation/menu pairs and no conflicting teacher labels. The 67 unique pairs comprise 15 next-goal, 16 activation, 23 reason, two finish and 11 act examples. Existing terminal-credit updates did not fit these teacher labels directly; these counts support a separately declared demonstration-fitting experiment, not a result of this continuation.

Limits: one seed, one six-edge public-root family, no derived goals, and the original 12 development evaluation cases reused. Random credit uses the controller RNG, so task schedules and warm state match while later exploration draws may differ. Full mission-boundary parameters plus frozen-source intra-mission assertions establish terminal-only updates; per-decision weight arrays and independent training-gradient reproduction are absent. Frozen evaluations are checked for no updates and no semantic changes. No untouched confirmation or general learning claim follows from this record.
