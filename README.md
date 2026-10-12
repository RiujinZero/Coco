# Coco

A native implementation of the "Laminar 200k Organism": a sparse, event-driven population of up to
200,000 identical six-layer INT4 units that learns online with local plasticity.

> **Unmaintained.** This is a research prototype released as-is. It is no longer developed or
> supported; issues and pull requests are not monitored.

## What it is

Units are addressed by learned lock/key tumblers, information dies early when it is not useful,
routing is motif-local, plasticity uses local accumulators, and structure grows by
consequence-driven splitting. The design is described in
[Laminar_200k_Organism_Design_Doc.docx](Laminar_200k_Organism_Design_Doc.docx) and the native
implementation follows [CPP_OPTIMIZATION_RUNBOOK.md](CPP_OPTIMIZATION_RUNBOOK.md).

Two implementations of the same organism live here and are checked against each other:

- `cpp/`: the native pipeline, one C++ call per event block with no Python in the hot path.
- `python/laminar_ref/`: the NumPy reference kernel.

On top of the organism the repository contains a structured continual learner (`laminar_train`)
with grounded dialogue, relational control and goal-directed training experiments, and a curriculum
built from public datasets.

## Layout

| Path | Contents |
|---|---|
| `cpp/include/laminar/`, `cpp/src/` | The organism, learning rules and continual-training profiles |
| `cpp/src/gpu/` | GPU port |
| `cpp/apps/` | Executables: trainers, probes and benchmarks |
| `cpp/tests/` | Native tests |
| `python/laminar_ref/` | NumPy reference implementation |
| `python/curriculum/` | Curriculum dataset builders (CLEVR, SQuAD, SmolTalk, TextWorld and others) |
| `python/` | Stream builders, parity checks, probes and analysis scripts |
| `scripts/` | Build, data-fetch, training and experiment scripts |
| `docs/` | Architecture and experiment documents |

## Documentation

- [Architecture](docs/ARCHITECTURE.md): maps the design document's requirements to the code
- [Canonical trainer](docs/CANONICAL_TRAINER.md)
- [Curriculum](docs/COCO_CURRICULUM.md), [curriculum dataset](docs/CURRICULUM_DATASET.md) and the [public data curriculum](COCO_PUBLIC_DATA_CURRICULUM.md)
- [Model task learner](docs/MODEL_TASK_LEARNER.md)
- [GPU port handoff](docs/GPU_PORT_HANDOFF.md)

[README_ORIGINAL.md](README_ORIGINAL.md) is the working README from development, with detailed run
commands and status notes. It and other documents may refer to machine-specific paths, logs and
validation files that are not part of this release.

## Build

Linux or WSL, with GCC (uses CMake if present, otherwise calls `g++` directly):

```bash
./scripts/build_wsl.sh
```

Windows, with MSVC 2022 Build Tools:

```bash
powershell -ExecutionPolicy Bypass -File scripts/build_msvc.ps1
```

Binaries land in `build-wsl/` or `build-msvc/`.

## Run

Apps take `key=value` parameters; see `Params::from_args` in `cpp/include/laminar/config.h`.

```bash
./scripts/fetch_tinystories.sh
```

```bash
./build-wsl/laminar_synth units=200000 train_events=120000 eval_events=16000
```

```bash
bash scripts/train_coco.sh --check
```

## What is not included

Training data, logs, checkpoints and validation result files are not part of this repository. The
builders in `python/curriculum/` and `scripts/` fetch and prepare the public datasets.

## License

Released under the [PolyForm Noncommercial License 1.0.0](LICENSE.md). You may use, modify and
share this code for personal, research and other noncommercial purposes. Commercial use is not
permitted.

Required Notice: Copyright (c) 2026 RiujinZero (https://github.com/RiujinZero/Coco)
