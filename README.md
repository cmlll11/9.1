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
