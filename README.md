# ProteinMutPD

ProteinMutPD distills a protein language model (the **teacher**) into a smaller one (the **student**) and tests the student on
[ProteinGym](https://proteingym.org) and [VenusMutHub](https://huggingface.co/datasets/AI4Protein/VenusMutHub).
You choose the teacher, the student, the training data, the evaluation data and the method on the command line, and one command
submits the whole chain of jobs to a Slurm cluster. Profiles for TAMU HPRC **Grace** and Vision are included;
[another cluster](docs/CLUSTERS.md) needs one YAML file. The trained student is a small LoRA adapter on a pretrained ESM2 or SaProt model.

The command-line program this repository installs is called `mpd`.

## Run a distillation

```bash
mpd submit --cluster grace --account YOUR_ACCOUNT --teacher esm2-3b --student esm2-650m --method score-all --train-data proteingym --eval-data proteingym venusmuthub
```

| option | what it chooses | values |
|---|---|---|
| `--teacher`, `--student` | the models | `esm2-8m`, `esm2-35m`, `esm2-150m`, `esm2-650m`, `esm2-3b`, `saprot-35m`, `saprot-650m` (`mpd models` lists them) |
| `--method` | how the student learns from the teacher | `direct-all`, `direct-one`, `score-all`, `score-one` (next table) |
| `--train-data` | the benchmark whose sequences the student trains on | `proteingym` or `venusmuthub` |
| `--eval-data` | the benchmark(s) to evaluate on | `proteingym`, `venusmuthub`, or both |
| `--account` | your Grace account (`myproject` prints it) | a number |

| | all assays | one assay |
|---|---|---|
| **direct logits**: the student copies the teacher's amino-acid logits at masked positions | `direct-all` | `direct-one` |
| **score**: the student copies the teacher's mutant-minus-wild-type score | `score-all` | `score-one` |

`*-all` trains one student on all assays. `*-one` trains one student per assay: add `--assay ID [ID ...]` (`mpd assays` lists the ids),
and expect one training per assay. When teacher and student come from different families (SaProt with ESM2), use `score-*`
([why](docs/SAPROT.md#a-teacher-and-a-student-from-different-families)). SaProt reads protein structures; for VenusMutHub it needs a
structure set that is not published yet ([docs/SAPROT.md](docs/SAPROT.md)).

`mpd submit` prints the plan (every job, its GPUs, hours and an upper bound in service units) and asks `Submit N job(s), up to X SU? [y/N]`.
`mpd plan` takes the same options and only prints the plan. Add `--dry-run` to `mpd submit` to write and check the job scripts without
queueing anything.

## Set up once

On a Grace login node, with the TAMU VPN if you are off campus and an allocation that `myproject` shows. Compute nodes have no
internet, so the Python environment, the data and the model weights are installed first. They live in `$SCRATCH/mpd`, not in
`$HOME` (10 GB), and each `fetch` can be run again if it is interrupted.

```bash
git clone https://github.com/yh79-ss/ProteinMutPD && cd ProteinMutPD
bash scripts/setup_env.sh --cluster grace --yes   # builds the Python environment in $SCRATCH/mpd/env (a few GB to download)
source $SCRATCH/mpd/env/bin/activate              # do this in every new shell
mpd fetch proteingym                              # the benchmark data
mpd fetch venusmuthub
mpd fetch model esm2-3b esm2-650m                 # the models you will use (about 14 GB for these two)
mpd doctor                                        # checks that everything is in place and says what to run if not
```

Then run the command above. If something the jobs need is missing, `mpd plan` and `mpd submit` say what it is and which command
installs it. [docs/GRACE.md](docs/GRACE.md) covers accounts, quotas and costs, and has a cheap first run to try before an expensive one.

## What happens after you submit

Slurm runs these jobs, each starting when the ones it needs have finished:

1. **prepare** (CPU) builds the table of masked positions and mutations from the data, and checks that the weights are there.
2. **teacher** (GPU) runs the teacher once and saves its outputs on disk.
3. **train** (GPU) trains the student's LoRA adapter to match those outputs. It saves its progress, so a job that hits its time limit
   continues where it stopped.
4. **score** (GPU) scores every variant of the evaluation data with the original student and with the trained one.
5. **evaluate** (CPU) computes the Spearman correlation for each assay and the average, for the original and for the trained student.

The hours and service units in the plan are estimates (upper bounds), not measurements. Follow the jobs with:

```bash
mpd status                       # your campaigns; with a campaign id, the state of its jobs
mpd logs CAMPAIGN train --err    # the log of a stage; without a stage it lists the logs
mpd results CAMPAIGN             # after evaluate: the original and the trained student side by side
mpd cancel CAMPAIGN              # stop a campaign
```

`CAMPAIGN` is the name `mpd submit` and `mpd status` print. Everything is written to `$SCRATCH/mpd/runs/CAMPAIGN/`: the numbers in
`evaluate/` and the trained adapter in `train/`. Scratch is not backed up, so copy what you want to keep ([docs/RESULTS.md](docs/RESULTS.md)
lists every file and metric).

A job that times out or lands on a bad node: repeat the same `mpd submit` command with `--force` (and a longer `--train-time` after a
time-out). Jobs that finished are kept. [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) maps messages to fixes.

## Use the trained student

A trained student is a PEFT LoRA adapter (`adapter_config.json` and `adapter_model.safetensors`) in
`<campaign>/train/<unit>/checkpoint_<update>/`; `mpd results` prints the path. Load it on the base model from `mpd fetch model`:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForMaskedLM, AutoTokenizer

base = "/scratch/user/<netid>/mpd/models/esm2-650m"        # the folder `mpd fetch model` made
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForMaskedLM.from_pretrained(base, torch_dtype=torch.float32, attn_implementation="eager")
model = PeftModel.from_pretrained(model, "<campaign>/train/all/checkpoint_02103").eval()   # bfloat16 on a GPU, as the jobs do
```

[docs/RESULTS.md](docs/RESULTS.md) scores a variant the way `mpd` does and explains the result files and metrics.

## Documentation

[GRACE](docs/GRACE.md) | [TROUBLESHOOTING](docs/TROUBLESHOOTING.md) | [RESULTS](docs/RESULTS.md) | [MODELS](docs/MODELS.md) |
[SAPROT](docs/SAPROT.md) | [DATA](docs/DATA.md) | [CLUSTERS](docs/CLUSTERS.md) | [ADDING_A_MODEL](docs/ADDING_A_MODEL.md) |
[LICENSES](docs/LICENSES.md) | [ZENODO](docs/ZENODO.md)

The code is MIT-licensed ([LICENSE](LICENSE)); the data and models keep their own licences, listed in
[docs/LICENSES.md](docs/LICENSES.md). To cite the software, use [CITATION.cff](CITATION.cff). Contributing: [CONTRIBUTING.md](CONTRIBUTING.md);
changes: [CHANGELOG.md](CHANGELOG.md); security: [SECURITY.md](SECURITY.md). For an issue, paste the output of `mpd doctor --report`,
which removes account numbers, user names and home paths.
