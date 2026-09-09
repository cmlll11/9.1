# Hard-sample GAP backdoor probe

This repository is the minimal implementation of the current CIFAR-10
verification experiment. It trains paired clean and BackdoorBench classifiers,
fits the official image-dependent `x + f(x)` GAP generator, selects hard
samples, and compares targeted attacks on clean and backdoored models.

The current protocol uses disjoint data partitions:

- 3,000 training images are reserved only for hard-sample selection;
- 47,000 training images are used to train every classifier and GAP generator;
- candidate hard samples are selected from the held-out 3,000-image partition.

The experiment is launched on the GPU server through:

```bash
env PYTHON_BIN=/path/to/python GPU_ID=0 DATA_ROOT=/path/to/cifar10 \
  bash bash/run_hard_sample_gap.sh
```

The launcher trains 10 clean classifiers, 20 official BackdoorBench
classifiers (BadNet, LF, Blended, and WaNet), and one x+f GAP generator for
each model. It then selects 100 samples that all five selection-clean
generators fail to move to target class 0 and evaluates them on the remaining
clean and qualified backdoor models.

The main outputs are written locally and ignored by Git:

- `artifacts/models/hard_sample_gap/`
- `artifacts/mappings/hard_sample_gap/`
- `artifacts/hard_samples/epsilon4/`
- `reports/hard_sample_gap_model_gates.json`
- `reports/hard_sample_gap_summary.json`
- `reports/hard_sample_gap_per_model.csv`

The official dependencies are kept as Git submodules under
`third_party/BackdoorBench` and `third_party/GAP`. Clone with
`--recurse-submodules`.

## Stage 1A/1B PGD-Probe pilot

The new pilot reuses existing checkpoints and does not train GAP generators.
Stage 1A uses Clean seeds 0 and 1 with Blended and WaNet seeds 0 and 1. It
searches CIFAR-100 with a coarse targeted-PGD epsilon grid, refines the Clean
Top-40 using 100-step PGD with three restarts, and compares the resulting
Top-30 samples on paired Clean/Backdoor models. Stage 1B performs within-model
and leave-one-Clean-model-out Ridge Probe evaluation on Clean seeds 0--2.

Run Stage 1A on the GPU server:

```bash
cd /path/to/9.1
PYTHON_BIN=/path/to/venv/bin/python \
DATA_ROOT=/path/to/data \
MODEL_ROOT=/path/to/9.1/artifacts/models/hard_sample_gap \
GPU_ID=0 \
bash bash/run_stage1a_bridge.sh
```

After Stage 1A, set `STAGE1A_RUN_DIR` to its result directory and run Stage 1B:

```bash
cd /path/to/9.1
STAGE1A_RUN_DIR=/path/to/9.1/results/stage1a_bridge/stage1a_YYYYMMDDTHHMMSSZ \
PYTHON_BIN=/path/to/venv/bin/python \
DATA_ROOT=/path/to/data \
MODEL_ROOT=/path/to/9.1/artifacts/models/hard_sample_gap \
GPU_ID=0 \
bash bash/run_stage1b_probe.sh
```

Stage 1B requires the existing partition metadata at
`$DATA_ROOT/hard_sample_gap/shared/partition.json` for Stage 1B, because the Probe
labels intentionally come from each model's known training samples. Results
are written under `results/stage1a_bridge/` and `results/stage1b_probe/`; no
existing run directory is overwritten.

## CIFAR-100 Probe transfer experiment

The transfer experiment trains a Ridge Probe from 1,000 CIFAR-100 train images
using Clean seeds 3 and 4, then applies it to a disjoint 1,000-image CIFAR-100
test pool on target Clean seeds 0--2. It compares Probe Top-100,
target-margin Top-100, and a refined-PGD reference obtained from coarse Top-300
selection. The primary Clean-only selector comparison is followed by same-seed
Clean versus BadNet, LF, Blended, and WaNet evaluation.

Run it on the GPU server with:

```bash
cd /path/to/9.1
PYTHON_BIN=/path/to/venv/bin/python \
DATA_ROOT=/path/to/data \
MODEL_ROOT=/path/to/9.1/artifacts/models/hard_sample_gap \
BATCH_SIZE=64 \
GPU_ID=0 \
bash bash/run_probe_cifar100_transfer.sh
```

The fine grid is `0.5, 1, 1.5, 2, 3, 4, 8, 16, 32 / 255`. Results are written
under `results/stage1b_cifar100_transfer/`.

## Stage 1C target-free untargeted transfer experiment

Stage 1C trains a target-free Ridge Probe on CIFAR-100 train images with
Clean seeds 3 and 4.  It evaluates Probe, top-1/top-2 margin, and a refined
PGD reference on Clean seeds 0--2, then performs deployment-style independent
selection on Clean, BadNet, LF, Blended, and WaNet models.  Part B also uses a
fixed Random-100 set shared within each seed as the no-selection baseline.

Untargeted success means that the model prediction changes from its original
prediction; no CIFAR-10 target class is supplied.  The grid is
`0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 8, 16, 32 / 255`.  The strong pass on
`CoarseTop300` is cached and reused for the refined reference and the other
Part A selectors.  Results are written under
`results/stage1c_cifar100_untargeted/`.

Run it on the GPU server with:

```bash
cd /path/to/9.1
PYTHON_BIN=/home/cml/.conda/envs/mdl-uap/bin/python \
DATA_ROOT=/home/cml/8.11/data \
MODEL_ROOT=/home/cml/8.11/artifacts/models/hard_sample_gap \
BACKDOORBENCH_ROOT=/path/to/9.1/third_party/BackdoorBench \
BATCH_SIZE=64 \
GPU_ID=0 \
bash bash/run_probe_cifar100_untargeted.sh
```

An existing model-gate report can optionally be supplied through
`QUALITY_REPORT=/path/to/hard_sample_gap_model_gates.json`; its native trigger
ASR values are copied into `model_quality.csv` without adding trigger logic to
the untargeted experiment.

## Stage 1D CIFAR-100 Probe multi-backdoor trigger alignment

Stage 1D trains a target-0 Ridge Probe from CIFAR-100 train images evaluated
by Clean seeds 1--3. Clean seed 0 applies the frozen Probe to a disjoint
CIFAR-100 test pool and selects the shared Top-100 images; no Clean0 full-pool
Oracle or refined-PGD reference is run. The same images are evaluated on
Clean0 and seed-0 BadNet, Blended, WaNet, SSBA, Input-Aware, and Adaptive-Blend
models using targeted PGD to CIFAR-10 class 0.

Each trigger type uses its own official test-time transform. BadNet, Blended,
WaNet, SSBA, and Input-Aware use their corresponding BackdoorBench assets or
saved states. Adaptive-Blend requires its official backdoor-toolbox trigger
and configuration. If an official transform cannot generate a trigger for the
CIFAR-100 test image, PGD records are retained but trigger alignment is
marked unavailable; no synthetic fallback trigger is used.

The experiment reports trigger activation on the selected images, separate
Clean control alignment for every trigger type, first-success targeted PGD
radius, fixed low-budget endpoints, direction concentration, and a
shuffled-trigger control.

Run it on the GPU server with:

```bash
cd /path/to/9.1
PYTHON_BIN=/home/cml/.conda/envs/mdl-uap/bin/python \
DATA_ROOT=/home/cml/8.11/data \
MODEL_ROOT=/home/cml/8.11/artifacts/models/hard_sample_gap \
QUALITY_REPORT=/home/cml/8.11/reports/hard_sample_gap_model_gates.json \
BATCH_SIZE=64 \
GPU_ID=0 \
bash bash/run_stage1d_trigger_alignment.sh
```

Results are written to a unique directory under
`results/stage1d_trigger_alignment/`.
