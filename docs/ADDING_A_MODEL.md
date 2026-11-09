# Adding a model

`mpd` knows models from one table, `src/mpd/registry.py`, so the planner stays importable without torch. A model can be a teacher, a
student, or both, and one class plays either role. This page lists what adding one takes. It is written from the ESM2 code
(`students/esm2.py`), which is the sequence-only pattern, and the SaProt code (`students/saprot.py`, [SAPROT.md](SAPROT.md)), which
adds structure input. Everything described here was read from the code and, for the licence lock and the manifest, covered by the
existing tests; adding a model has not been rehearsed with a third family.

Two cases:

- **another checkpoint of a family that exists** (say a new ESM2 size): a registry row, a manifest, tests and docs (steps 0, 2, 4, 5 and 6);
- **a new family**: all steps.

## 0. Before anything: the licence

Read the weights' licence and the licence of the outputs. If the licence limits what the model's outputs may train, say so in the
registry (step 2, "Output-family locks") **before** the model can be planned; the lock holds after the model works. For the weights' own
licence, set `ModelInfo.license` (shown by `mpd models`) and add the model to [LICENSES.md](LICENSES.md). A model whose weights must
not be fetched or redistributed by `mpd` gets no manifest rows (step 4).

## 1. The surface a model must offer

The stages call these on the object that `students.load(key, device, lora=None, adapter=None, dtype=None)` returns
(`students/__init__.py`):

| member | called by | what it must do |
|---|---|---|
| `family` | informational | the family string of the registry row (`Esm2.family` is `"esm2"`) |
| `module` | all stages | the `torch.nn.Module`: `.train()`, `.eval()`, `.parameters()`, `.named_parameters()`. DDP wraps this |
| `encode(table, structures=None)` | teacher, train, score | tokenise the `ContextTable`'s sequences once (`table.sequences`); `structures` is a structure set for models that need one, ignored otherwise |
| `logits(table, rows, forward=None)` | all stages | for the contexts `rows` of the table, mask the focal residue (`table.pos`) and return the **raw logits of the 20 amino acids at that position, `[len(rows), 20]`, float32, in the order `ACDEFGHIKLMNPQRSTVWY`** (`constants.AA20`). `forward`, when given, replaces `module` for the forward pass (DDP) |
| `trainable()` | train | the list of parameters the optimizer updates (the LoRA matrices) |
| `save_adapter(path)` | train | write the trained adapter (PEFT format) to `path` |
| constructor `(key, device, dtype, *, lora, adapter)` | `students.load` | `lora={"rank": r, "alpha": a}` builds a trainable LoRA model; `adapter=<folder>` loads a trained one frozen; neither gives the frozen base model (the teacher, and the `base` student) |

Rules the existing classes follow, each there for a reason:

1. **Import torch, transformers and peft only inside this module** (and import this module lazily from `students/__init__.py`), so
   `mpd plan`, `submit`, `doctor` and `fetch` keep working without torch. A test enforces that the planner modules import no torch.
2. **Load weights through `mpd.data.models.resolve(key)`, never by Hub id.** Jobs are offline. `resolve` honours `MPD_MODEL_<KEY>` and the
   fetched folder and raises a message that names the fetch command; there is no silent fallback.
3. **Name the base model by its Hub id** (`model.name_or_path = model.config.name_or_path = info.hf_id`) and set the PEFT config's
   `revision` to the registry pin when the weights are the managed folder, so a saved adapter carries no local path.
4. **`save_adapter` passes `save_embedding_layers=False`**; otherwise, with the base model named by its Hub id, PEFT asks the Hub for a
   `config.json` and spends about 20 seconds on retries when the network is blocked.
5. **Dropout 0.** The recipe forces it (SaProt's configuration says 0.1). LoRA is rank 8, alpha 16, no LoRA dropout, on the attention
   projections (`LORA_TARGETS` in `esm2.py`); the preset carries the numbers.
6. **One token per residue**, checked in `encode`, and the 1,024-token window around the focal position for longer sequences
   (`esm2.optimal_window`, which `saprot.py` imports rather than copies).
7. **dtype.** `esm2.default_dtype()`: bfloat16 on a GPU with native support, float32 otherwise, `MPD_DTYPE` overrides.

A model whose natural output is not raw logits (SaProt gives log-marginals over 21 structure states) converts them inside `logits` so
that differences of the returned values equal the model's own mutation score, and says so in its page.

## 2. The registry row

In `src/mpd/registry.py`:

```python
ModelInfo("mymodel-35m", "myfamily", "someorg/MyModel_35M", 35, ("student", "teacher"), act_gb=3,
          revision="<40-hex commit>", license="MIT", status="planned", note="Planned: ...")
```

| field | meaning |
|---|---|
| `key` | lower case, letters, digits, `-`; what `--teacher` and `--student` take. The override variable is `MPD_MODEL_<KEY>` with the key in capitals and `-` as `_` |
| `family` | picks the student class in `students/__init__.py` and the preset in `presets.py` |
| `hf_id`, `revision` | the Hugging Face repository and the 40-hex commit the manifest was built from (step 4) |
| `params_m` | millions of parameters; sizes the walltime estimate (`(params / 655) ** 0.9`) and the weights part of the memory estimate |
| `roles` | subset of `("student", "teacher")` |
| `status` | `available` or `planned`. The planner rejects a planned model with its `note`. Keep `planned` until the tiny end-to-end test passes for the family |
| `needs_structure`, `needs_msa`, `shard_across_gpus` | what the inputs and the memory need; only `needs_structure` is implemented (SaProt) |
| `teacher_backends` | `("live",)`: the teacher stage runs the model. A `"cache"` backend would mean the teacher table is supplied from outside; no model uses it |
| `act_gb`, `act_gb_measured` | estimated activation memory in GB at micro-batch 2, which scales with the micro-batch. Count the tensors autograd saves for a 2 x 1,024-token batch (the ESM2 values were counted on the meta device, `tests/test_models.py::test_esm2_activation_estimates_follow_the_meta_device_accounting`), or measure on a GPU and set `act_gb_measured=True`. The planner uses it only to pick a GPU and a micro-batch |
| `license` | licence of the weights, shown by `mpd models` and recorded by `mpd fetch model` |
| `output_family_lock`, `license_note` | see below |
| `note` | one line `mpd models` prints under the row |

### Output-family locks

If a licence says the model's outputs may not be used to train other, similar models, set
`output_family_lock="<its own family>"` and `license_note` to the clause (wording as in `GENBIO_LICENSE_NOTE`). `RunSpec.validate`
then refuses the model as the **teacher** of any student of another family, with the licence text and a pointer to
[LICENSES.md](LICENSES.md). The check runs **before** the "planned" check, so it still holds once the model is implemented. There is no
override flag, on purpose: lifting a lock needs written permission from the licensor and a code change.

For such a model also: give it **no rows in `models.tsv`** (its state is `unpinned` and `mpd fetch model` refuses it, as for AIDO), ship
no outputs, caches or adapters made from it, and add a test like those in `tests/test_licenses.py` (the lock fires before the
"planned" message; only the intended model is locked; nothing of it is pinned). [LICENSES.md](LICENSES.md) says what `mpd` enforces and
why for AIDO, the one model that has a lock today.

## 3. The student module and the preset

1. Write `src/mpd/students/<family>.py` with a class that has the surface above, and add its branch to `students.load`:

   ```python
   if family == "myfamily":
       from .myfamily import MyModel
       return MyModel(key, device, dtype, lora=lora, adapter=adapter)
   ```

2. Presets are data in `src/mpd/presets.py`: `PRESETS[(family, objective)]` with `objective` in `direct_logits` and `score`. A family that
   trains like ESM2 reuses `_ESM_LIKE_DIRECT` and `_ESM_LIKE_SCORE` (that is what SaProt does). If you set other numbers, say where
   they come from in the file; the numbers there come from the earlier study's runs.
3. A model that needs inputs besides the sequence needs its own preparation in the CPU `prepare` stage, a record of those inputs in the
   campaign, and a preflight (`preflight_checks(spec, home, cluster=None)` returning `mpd.checks.Check` rows, as `mpd.data.models` and
   `mpd.structure` do; `run_checks` in `preflight.py` calls those two by name, so a third needs a call added there). Keep that code free
   of torch and out of the stage modules: only `stages/evaluate.py` may open labels (`tests/test_stages.py` scans the other stages for
   `.labels(`). Follow `mpd/structure/`.

## 4. The weights manifest

`src/mpd/data/models.tsv` lists every file of every fetchable model with its size and sha256; `mpd fetch model` verifies against it and
`mpd submit` refuses to queue a run whose weights are not `ready`. Do not edit it by hand. With the registry row in place (it may
start without a `revision`; the tool prints the one to paste):

```bash
python scripts/maintainers/pin_models.py --models mymodel-35m --at main --write
```

rebuilds the rows for that model at the repository's current `main`, writes the manifest and prints the `revision=` value for the
registry row. It reads the Hugging Face API anonymously (no token), one request at a time, and keeps `config.json`, the three tokenizer
files (`vocab.txt`, `special_tokens_map.json`, `tokenizer_config.json`) and the first weights format that exists (`model.safetensors`,
sharded safetensors, `pytorch_model.bin`, sharded `.bin`); it stops with a message if a repository lacks one of them. Afterwards
`python scripts/maintainers/pin_models.py` (exit status 0) checks that the manifest and the registry agree, and `--crosscheck` compares
each large file with a second endpoint. Pickle-only weights (`.bin`) need torch 2.6 or newer; the preflight warns about it.

## 5. Tests to copy

All run on CPU without a network or a GPU.

| to prove | copy from |
|---|---|
| a tiny random model of your family, so no test needs a download | `make_tiny_esm` in `tests/tinydata.py`; `make_tiny_saprot` and the fixtures in `tests/fx_saprot.py` |
| tokens, focal masking, window, the 20 logits, adapter round trip, `students.load` dispatch | `tests/test_saprot_student.py` |
| the manifest and the registry agree; every available model can be fetched; the docs page agrees with the code | `tests/test_models.py` (`test_manifest_has_the_documented_shape`, `test_every_pinned_model_is_in_the_registry_with_a_40_hex_commit`, `test_every_available_model_can_be_fetched`, the docs-page tests) |
| a licence lock | `tests/test_licenses.py` |
| the whole pipeline on the tiny model, including "a teacher identical to the student starts at zero loss" for both objectives | `tests/test_stages.py` with `tests/fx_stages.py` |

Run a single file the way the contributing guide says; the pipeline test is the one that decides step 6.

## 6. Switch it on, and say so

1. Change `status` to `available` only when the pipeline test above passes for the family. Until then the `note` explains what is
   missing.
2. Add a row to the table in [MODELS.md](MODELS.md) (download size from `mpd fetch model --list`), to [LICENSES.md](LICENSES.md), to the
   README model table, and, for special inputs, a page like [SAPROT.md](SAPROT.md).
3. Say plainly what has been verified: tested on CPU with a tiny model; compared with recorded numbers (which, how close); not run on a
   GPU or on Grace. Nobody should have to guess.
