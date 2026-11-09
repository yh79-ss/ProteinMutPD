# Results: what a campaign leaves behind, what the numbers mean, and how to use the student

Tags as in [GRACE.md](GRACE.md): **[CPU]** tested on CPU here, **[NOT VERIFIED]** nobody has run it. The listing, the file excerpts and
the `mpd results` output below come from a campaign that was run **on CPU with tiny random models and synthetic labels** (three made-up
assays), through the real stages: the numbers mean nothing, the shapes are real. No campaign has run on a GPU or on Grace.

## What is in a campaign directory

`<home>/runs/<campaign id>/`. The id is `<teacher>__<student>__<train data>-<method>[__<assays>]__seed<N>[__<recipe hash>]__<8 hex>`; the last
group hashes the options that can change the results or the layout of the results: evaluation data, cluster, GPU type, training GPU count,
score shards, and the numerics variables `MPD_DTYPE`, `MPD_TF32` and `MPD_STRUCTURE_LEGACY_MASK`. So the same recipe with another `--eval-data`
is a different campaign. Walltimes, `--array-throttle`, `--sbatch-arg`, `--module`, other `--env` pairs, the account, `--home` and `--python`
are not part of the id: after a TIMEOUT or a bad node the recovery stays in the same campaign and finds the finished stages
([GRACE.md](GRACE.md#when-a-job-does-not-finish)).

```text
contexts/            the masked positions every stage reads (union, per unit, per evaluation dataset), MANIFEST.json; no labels
                     (a SaProt campaign adds structures.json, the structure strings that `prepare` froze)
teacher/             logits.npy (the teacher's 20-amino-acid logits per training context), COMPLETE.json
train/<unit>/        CONFIG.json  COMPLETE.json  SELECTED.json  dev_curve.jsonl  train_steps.csv
                     checkpoint_<update>/   adapter_config.json  adapter_model.safetensors  README.md
score/<dataset>/<unit>/   shard_<i>_of_<n>.npz and .json (the student's logits for one shard of the evaluation contexts)
evaluate/            summary.json  per_assay.csv
spec.json  resolved.json  plan.json  tasks/<stage>.json  scripts/<stage>.sbatch
logs/  SUBMITTED.json     (added by mpd submit: the Slurm output and error files, and the job ids)
```

Listing of the tiny test campaign **[CPU]** (`logs/` and `SUBMITTED.json` are absent because it ran without Slurm; 79 KB in all):

```text
contexts/{eval__proteingym,union,unit__all}.{json,npz}  contexts/MANIFEST.json
teacher/COMPLETE.json  teacher/logits.npy
train/all/{CONFIG,COMPLETE,SELECTED}.json  train/all/{dev_curve.jsonl,train_steps.csv}
train/all/checkpoint_00006/{adapter_config.json,adapter_model.safetensors,README.md}
score/proteingym/{base,all}/shard_00{0,1}_of_002.{json,npz}
evaluate/{per_assay.csv,summary.json}
spec.json  resolved.json  plan.json  scripts/*.sbatch (6)  tasks/*.json (6)
```

- `<unit>` is `all` for the `*-all` methods and the assay id for `*-one`. `base` is the frozen student: scored, never trained.
- **Only the selected update's adapter is kept**: of the three monitored updates, the one with the lowest development loss, preferring
  updates that beat the untrained start; `MPD_KEEP_ALL_CHECKPOINTS=1` keeps all three. `SELECTED.json` names the update, and
  `mpd results` prints the folder.
- `train/<unit>.partial/` is the resumable state while a unit is unfinished (about 32 MB for a 650M student, 71 MB for a 3B student;
  deleted on completion). `train/<unit>.incomplete-<epoch>/` is an earlier attempt that could not be resumed: delete it when you no
  longer need it. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#timeout-the-walltime-was-too-short).
- A submitted campaign's recipe files (`spec.json`, `resolved.json`, `tasks/`) are never rewritten, whatever state its jobs are in; only the scripts
  and `plan.json` are, on `--force` or when a cancelled or failed campaign is submitted again. A command that would write a different recipe into the
  directory is refused ([GRACE.md](GRACE.md#a-different-recipe-is-refused)).
- Every record the stages write (`teacher/COMPLETE.json`, `train/<unit>/CONFIG.json` and `COMPLETE.json`, each score shard `.json`,
  `evaluate/summary.json`) has a `provenance` block (mpd, torch, transformers, peft, Python, host, Slurm job id, time; on a GPU node the
  GPU and driver). It has no paths and can go into an issue.
- A campaign directory is small. The tiny one is 79 KB; for a 650M student the adapter is about 11 MB (2.7 million float32 LoRA
  parameters), the logit tables are a few MB per unit and dataset, and the `*-one` methods write a few thousand small files over many
  assays. These are estimates from array sizes, **[NOT VERIFIED]** on a real run.

## The result files

### `evaluate/per_assay.csv`

One row per dataset, unit, assay and readout. `spearman` is `nan` when it is undefined (constant predictions, or fewer than three variants).

```text
dataset,unit,assay_id,readout,n_variants,spearman
proteingym,base,ASSAY_A,delta,228,0.02841718497039012
proteingym,base,ASSAY_A,dual,228,0.01816079548273373
proteingym,base,ASSAY_B,delta,228,-0.10093684473770272
```

### `evaluate/summary.json`

| key | meaning |
|---|---|
| `campaign` | the campaign id |
| `objective` | `direct_logits` or `score` |
| `primary_readout`, `primary_readout_basis` | `delta` for direct-logit runs and `dual` for score runs, and the sentence that says this is an **assumption, not a confirmed convention** (below) |
| `eval_data`, `units` | the evaluated datasets; `["base", *training units]` |
| `assays_in_dataset` | `{dataset: number of assays}`: the coverage reference for `assays` below |
| `results` | one entry per dataset, unit and readout (below) |
| `dms_labels_opened` | `true`: only this stage reads labels (a test enforces it) |
| `provenance` | as above |

An entry of `results`: `dataset`, `unit`, `readout`, `assays` (assays evaluated for that unit and dataset), `assays_finite` (how many had a
finite Spearman; the macro averages only those), `macro_spearman`, `primary` (true if this readout is the primary one), and for a trained
unit `base_macro_same_assays` and `delta_vs_base`:

```json
{"assays": 3, "assays_finite": 3, "dataset": "proteingym", "unit": "all", "readout": "dual", "primary": true,
 "macro_spearman": -0.057323730739630464, "base_macro_same_assays": -0.057455242608868175, "delta_vs_base": 0.0001315118692377104}
```

### `mpd results <campaign>`

Prints the same numbers as tables. The output for the tiny campaign **[CPU]** (random models, so the correlations are noise):

```text
campaign   esm2-35m__esm2-8m__proteingym-score-all__seed20260922__bb5ef21c__c2c58eb6
objective  score   readouts: delta and dual, both always reported

frozen student (before training), macro Spearman over every assay of the dataset
dataset     readout  assays  macro Spearman
proteingym  delta         3         -0.0526
proteingym  dual *        3         -0.0575

distilled student against the same student before training, on the same assays
dataset     readout  unit  assays     base  distilled  delta_vs_base
proteingym  delta    all        3  -0.0526    -0.0525        +0.0001
proteingym  dual *   all        3  -0.0575    -0.0573        +0.0001

* primary readout by this tool's convention (delta for direct-logit runs, dual for score runs). That convention is an assumption of the maintainer, not a confirmed standard; both readouts are always reported.
```

It then lists the two result files, the adapter folder of each unit, and, for a Grace campaign, a reminder that scratch is not backed up with an `rsync` line (a campaign of another cluster gets a plain reminder to copy what you keep).
With more than 12 units (`*-one` over many assays) it prints one row per dataset and readout (`mean of N units (K improved)`); the per-unit values
stay in `per_assay.csv`. Before the `evaluate` stage has finished it says `no results yet` and exits with status 1.

## What the metrics mean

- **Spearman per assay.** The rank correlation, within one assay, between a variant score and the measured value. Variants are the
  assay's single substitutions.
- **Macro Spearman** is the mean over assays of that value, over the assays where it is finite (`assays_finite` of `assays`). It is
  not pooled over variants.
- **Two readouts, always both.** For every variant the student's 20 amino-acid logits `s` at the masked position give

  | readout | score |
  |---|---|
  | `delta` | `s[mutant] - s[wild type]`, the zero-shot masked-marginal convention |
  | `dual` | `log softmax(s / 1.0)[mutant] - log softmax(s / 1.5)[wild type]`, the readout the `score` methods train |

- **`delta_vs_base`** is `macro_spearman - base_macro_same_assays`: the distilled student against the same student before training, on
  exactly the same assays (for a one-assay unit that is its own assay). It is the cleanest single number, because both sides went
  through the same pipeline.
- **`base`** is evaluated on every assay of the dataset. A unit trained on one assay is evaluated on that assay only when the
  evaluation dataset is the training dataset.
- **Which readout is primary.** `primary` marks `delta` for direct-logit runs and `dual` for score runs. That pairing is an **unconfirmed
  assumption of the maintainer**, not a convention anyone checked against a reference; read both readouts, and say which one you
  report.

## Base-model reference values

If your student is `esm2-650m`, the `base` unit is the frozen ESM2-650M, and the earlier study recorded its outputs as
mutant-minus-wild-type logits, which is the `delta` readout. Replaying those stored outputs (no model run, on CPU) through this
repository's loaders and evaluator gave ([DATA.md](DATA.md#regression-against-the-earlier-study)):

| benchmark | macro Spearman, this code | recorded |
|---|---|---|
| VenusMutHub, 793 assays | 0.2095016689 | 0.2095016689 |
| ProteinGym, 217 assays | 0.4279547 | 0.4279488 |

The ProteinGym difference of 6e-6 is a tie effect: the earlier evaluator stored labels as float32, and with labels rounded the same way all
217 assays are equal (0.4279487815). Use these as a sanity check of a first real run: the `delta` rows of `base` should agree to a few thousandths. Running
ESM2-650M on CPU in float32 on six short assays reproduced the recorded per-assay Spearman to within 0.003 (the earlier run was bf16 on a
GPU). A whole-benchmark run on a GPU has **not** been compared **[NOT VERIFIED]**; a larger gap means a data or code difference worth reporting.

## Comparing with the earlier study

| | status |
|---|---|
| ESM2-650M base numbers on both benchmarks | reproduced from stored outputs, as above |
| The VenusMutHub cohort (793 assays, 489 proteins, 25,044 rows) | reproduced exactly, labels equal |
| A **distilled** student's numbers | **not reproduced.** The pipeline runs on CPU with tiny models; no full training has run on a GPU, so the first real run is the first comparison. Differences from bf16 and GPU non-determinism are expected |
| Cells whose teacher was AIDO (the earlier study's headline cells) | **cannot be reproduced with `mpd`**: outputs of AIDO may not train ESM2 or SaProt under its licence, and `mpd` refuses the run ([LICENSES.md](LICENSES.md)) |
| SaProt on ProteinGym | the official AF2 structures differ from the ones used earlier (none of the 217 files is identical), so earlier SaProt numbers are a reference to compare in spirit, not to reproduce ([DATA.md](DATA.md), [SAPROT.md](SAPROT.md)) |
| SaProt on VenusMutHub | the earlier numbers contain a masking artefact for the 98 ESMFold proteins (154 of 793 assays); `mpd` masks correctly, so those cells will differ ([SAPROT.md](SAPROT.md#the-esmfold-scale-artefact-in-the-earlier-studys-recorded-saprot-numbers)) |

Compare like with like: the same dataset, the same assays, the same readout, and `delta_vs_base` before absolute values.

## Using the trained student

The adapter is PEFT LoRA: rank 8, alpha 16, on the query, key, value and attention-output projections of the base model; `adapter_model.safetensors`
holds only the LoRA matrices. `adapter_config.json` names the base model by its Hugging Face id (`facebook/esm2_t33_650M_UR50D`), not by a
local path, and records the pinned commit as `revision` when the weights came from `mpd fetch model` **[CPU]**, so the folder can be copied
anywhere. Load it on the base model, from the folder `mpd fetch model` made or from the Hub on a machine with internet. This runs on CPU and
reproduces `mpd`'s own logits and both readouts exactly in float32 on the tiny model **[CPU]**:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForMaskedLM, AutoTokenizer

base_dir = "/scratch/user/<netid>/mpd/models/esm2-650m"        # the folder `mpd fetch model` made
adapter_dir = "<campaign>/train/all/checkpoint_02103"          # `mpd results <campaign>` prints the folder

tok = AutoTokenizer.from_pretrained(base_dir)
model = AutoModelForMaskedLM.from_pretrained(base_dir, torch_dtype=torch.float32, attn_implementation="eager")
model = PeftModel.from_pretrained(model, adapter_dir)
model.eval()

AA20 = "ACDEFGHIKLMNPQRSTVWY"
aa_ids = tok.convert_tokens_to_ids(list(AA20))


def masked_logits(sequence: str, position: int) -> torch.Tensor:
    """Raw logits of the 20 amino acids at `position` (0-based), in AA20 order. Up to 1,022 residues."""
    ids = tok(sequence, return_tensors="pt")["input_ids"]
    ids[0, position + 1] = tok.mask_token_id               # +1 because <cls> comes first
    with torch.no_grad():
        return model(input_ids=ids).logits[0, position + 1][aa_ids]


def delta(sequence: str, position: int, mutant: str) -> float:
    """The `delta` readout: s[mutant] - s[wild type] at the masked position."""
    z = masked_logits(sequence, position)
    return float(z[AA20.index(mutant)] - z[AA20.index(sequence[position])])


def dual(sequence: str, position: int, mutant: str) -> float:
    """The `dual` readout: log softmax(s / 1.0)[mutant] - log softmax(s / 1.5)[wild type]."""
    z = masked_logits(sequence, position)
    top = torch.log_softmax(z / 1.0, dim=-1)[AA20.index(mutant)]
    bottom = torch.log_softmax(z / 1.5, dim=-1)[AA20.index(sequence[position])]
    return float(top - bottom)
```

Notes: the jobs use bfloat16 on a GPU with native support (`torch_dtype=torch.bfloat16` and `.to("cuda")` to match) and float32 otherwise, so
float32 on a laptop differs slightly from the numbers in `per_assay.csv`. For a sequence longer than 1,022 residues `mpd` scores a
1,024-token window centred on the position (`mpd.students.esm2.optimal_window`); this snippet does not. To rebuild the table of a
whole assay, take `mutated_sequence` and `DMS_score` from its CSV and correlate with `scipy.stats.spearmanr`. The same recipe works for
`saprot-*` students only with their structure input, which is described in [SAPROT.md](SAPROT.md).

## Copying results off scratch

`$SCRATCH` is not backed up and its purge rule is not documented **[NOT VERIFIED]**. From your own computer (this is the line
`mpd results` prints, plus the adapter):

```bash
rsync -av <netid>@grace.hprc.tamu.edu:/scratch/user/<netid>/mpd/runs/<campaign>/{evaluate,spec.json,resolved.json} ./<campaign>/
mkdir -p ./<campaign>/train/all
rsync -av <netid>@grace.hprc.tamu.edu:/scratch/user/<netid>/mpd/runs/<campaign>/train/all/checkpoint_02103 ./<campaign>/train/all/
```

Keep `evaluate/` (the numbers), `spec.json` and `resolved.json` (the recipe), the `provenance` blocks (they are in those files) and the adapter
folder of each unit you care about. The teacher table, the score shards and the contexts can be regenerated and are the bulk of the
files; delete them when the numbers are safe. HPRC's portal and Globus also copy files off Grace. If you publish an adapter, read
the licences of the base model and of the data first ([LICENSES.md](LICENSES.md); not legal advice).
