# Model weights

Jobs run offline: HPRC's documentation says Grace compute nodes have no outbound internet unless a job loads the
`WebProxy` module (not verified by this project), and `mpd` jobs set `HF_HUB_OFFLINE=1`. The weights therefore have to
be on disk before you submit. `mpd fetch model` puts them there, once, on a login node. Each model is pinned to one
Hugging Face commit and every file is checked against a recorded size and sha256, so a run uses the same bytes on every
cluster.

## Quick start

```bash
mpd fetch model --list --cluster grace                       # the pinned models, their size, and what you already have
mpd fetch model esm2-3b esm2-650m --cluster grace            # 13.98 GB, resumable, login node only
mpd doctor --cluster grace                                   # the "weights" line says what is ready
```

`esm2-3b` and `esm2-650m` are the teacher and student of the README example. A run needs only the two models it names.
`mpd submit` refuses to queue a job while one of them is not ready, and prints the command that fixes it.

## The models

| key | Hugging Face repository | pinned commit | download | weights file | licence |
|---|---|---|---|---|---|
| `esm2-8m` | `facebook/esm2_t6_8M_UR50D` | `c731040f` | 31.4 MB | safetensors | MIT |
| `esm2-35m` | `facebook/esm2_t12_35M_UR50D` | `6fbf070e` | 136.0 MB | safetensors | MIT |
| `esm2-150m` | `facebook/esm2_t30_150M_UR50D` | `a695f604` | 595.3 MB | safetensors | MIT |
| `esm2-650m` | `facebook/esm2_t33_650M_UR50D` | `08e4846e` | 2.61 GB | safetensors | MIT |
| `esm2-3b` | `facebook/esm2_t36_3B_UR50D` | `476b6399` | 11.37 GB | PyTorch pickle, 2 shards | MIT |
| `saprot-35m` | `westlake-repl/SaProt_35M_AF2` | `316cd401` | 136.9 MB | PyTorch pickle | MIT |
| `saprot-650m` | `westlake-repl/SaProt_650M_AF2` | `d9b9ad00` | 2.61 GB | PyTorch pickle | MIT |

All seven together are 17.48 GB; the five ESM2 models are 14.74 GB (`mpd fetch model --all` fetches every model the
pipeline can use today). The SaProt weights can be fetched already, but the pipeline cannot run them until the structure
inputs land (`mpd models` shows each model's status). The licences are the ones on the Hugging Face model cards, checked on
2026-11-09; the AIDO model is a different case, see [below](#aido-is-never-fetched). The full commits are in
`src/mpd/registry.py`, and the file names, sizes and sha256 in `src/mpd/data/models.tsv`:

```
esm2-8m      c731040fcd8d73dceaa04b0a8e6329b345b0f5df
esm2-35m     6fbf070e65b0b7291e7bbcd451118c216cff79d8
esm2-150m    a695f6045e2e32885fa60af20c13cb35398ce30c
esm2-650m    08e4846e537177426273712802403f7ba8261b6c
esm2-3b      476b639933c8baad5ad09a60ac1a87f987b656fc
saprot-35m   316cd4017d29f4657b959365f24b57f1ee278912
saprot-650m  d9b9ad00ef61c0990e611b2b43f2231c7de24b38
```

Only the files that `from_pretrained(<folder>)` needs are fetched: `config.json`, the three tokenizer files and the
weights. The duplicate `pytorch_model.bin` and `tf_model.h5` of the small ESM2 repositories, and SaProt's own `.pt`
checkpoints, are left out.

## Where the weights go

Weights live in plain folders with real files, `<home>/models/<key>/`, plus a `FETCHED.json` that `mpd` writes last.
They are not in a Hugging Face cache, so `HF_HOME` plays no part: a folder loads with no network request (checked on
CPU, with transformers 4.51.0 in the test suite and 4.57.6 while designing this). `<home>` is `--home`, else
`$MPD_HOME`, else the cluster's default (`$SCRATCH/mpd` on Grace). Keep it on `$SCRATCH`: `$HOME` is 10 GB and 10,000
files on Grace (HPRC), and `esm2-3b` alone is 11.37 GB.

Two variables change where `mpd` looks:

- `MPD_MODELS_DIR=/shared/models` replaces `<home>/models`, so a lab can keep one read-only copy for everyone.
- `MPD_MODEL_ESM2_650M=/path/to/folder` (the key in capitals, `-` as `_`) uses that folder for one model, whatever
  `<home>/models` holds. It must contain `config.json` and the files `from_pretrained` needs; `mpd` does not check them
  against the pin, so the result is your responsibility. This is for weights you already have under another name.

`MPD_ALLOW_HUB=1` lets a login node or laptop load a model by its Hugging Face id (at the pinned commit) when no local
copy exists. Jobs never get it: a missing folder in a job is an error, not a download.

## Fetching

```bash
mpd fetch model esm2-3b esm2-650m --cluster grace        # download what is missing, check every file, mark the folder
mpd fetch model esm2-3b --cluster grace --dry-run        # say what would be downloaded and whether there is room; write nothing
mpd fetch model esm2-3b --cluster grace --verify         # also re-hash the files that are already there
mpd fetch model esm2-3b --cluster grace --force          # download everything again
mpd fetch model --all --cluster grace                    # every model the pipeline can use
```

How it behaves:

- **One file at a time, anonymous.** No Hugging Face token is read or sent (all seven repositories are public). Each model is
  at most seven requests, 0.3 s apart, far inside the anonymous limits. On HTTP 429 or 503 `mpd` waits the `Retry-After` time
  and goes on; if the limit persists, run the same command later.
- **Resumable.** A connection that drops, or a login-node session that ends, leaves a `<file>.part` next to the model; the next run
  continues it with an HTTP Range request instead of starting over. A file that was cut short (an interrupted `rsync`) resumes
  the same way.
- **Checked.** Every file must match its pinned size and sha256 before it counts. A mismatch is retried once from byte 0, then the
  file is deleted and the fetch stops with `sha256 mismatch`. `FETCHED.json` appears only after all files pass.
- **Login-node limit.** Processes older than 60 minutes are killed on Grace login nodes (HPRC), so with a cluster profile
  (`--cluster` or `$MPD_CLUSTER`) the fetch stops by itself five minutes before that, keeps its partial files and exits
  with status 3. Run the same command again to continue. `--time-budget MIN` sets the limit yourself.
- **Room.** Before downloading, `mpd` checks that the filesystem has the missing bytes plus 1 GiB. The check cannot see your quota,
  so a write that hits it says `disk full or quota exceeded` and points to `showquota`.
- **One at a time per model.** A second `mpd fetch model` for the same model while one is running stops with `already writing`.
- **If the pinned commit ever disappears** from Hugging Face (history rewritten), `mpd` tries `main` for that file and accepts it only if
  the sha256 matches the pin. This was tested against a local test server only.

## No internet on the cluster

Fetch on any machine that has it (a laptop, another cluster) and copy the folders over. The fetch step needs only the base
install of `mpd`, not torch.

```bash
# on the machine with internet
mpd fetch model esm2-3b esm2-650m --home ./stage
rsync -av ./stage/models/ <netid>@grace.hprc.tamu.edu:/scratch/user/<netid>/mpd/models/     # or scp, or Globus

# on Grace: no download happens; this re-hashes every file and confirms the folders
mpd fetch model esm2-3b esm2-650m --cluster grace --verify
```

Or let `mpd` do the copying and the checking in one step on the cluster: `mpd fetch model esm2-3b esm2-650m --cluster grace --from-dir
./stage/models`. For each model `--from-dir DIR` takes the first of these that holds all its files: `DIR/<key>/`, `DIR/<repository
name>/` (for example `DIR/esm2_t33_650M_UR50D/`), a Hugging Face cache snapshot (`DIR/models--facebook--esm2_t33_650M_UR50D/snapshots/<commit>/`,
also under `DIR/hub/`, so `--from-dir ~/.cache/huggingface` works), and `DIR` itself. Symlinks are followed and real files are written.
An empty snapshot folder, or one whose symlinks are broken because the blobs were not copied, is reported as such.

If the files are already on the cluster under another name, either point `MPD_MODEL_ESM2_650M` at the folder (immediate, unchecked), or
use `--from-dir` to copy them into `<home>/models` and have them verified.

`HF_ENDPOINT=https://mirror.example` is honoured for a Hugging Face mirror; the sha256 check still applies.
Fetching from inside a Slurm job (for example after `module load WebProxy`) should work because the download honours
`https_proxy`, but this has not been tried.

## What state is a model in

`mpd models --cluster grace` and `mpd fetch model --list --cluster grace` show one word per model; `mpd doctor` explains it.
Looking at the state reads file sizes only, so it is instant on a login node; `--verify` hashes.

| state | meaning | fix |
|---|---|---|
| `ready` | every file has its pinned size, and `FETCHED.json` records this release's commit and hashes | none |
| `override` | `MPD_MODEL_<KEY>` points at a folder with a `config.json`; its contents are not checked | none |
| `missing` | nothing fetched yet (or the override points nowhere) | `mpd fetch model KEY` |
| `incomplete` | some files are absent, usually an interrupted fetch; a `.part` shows how far it got | the same command resumes |
| `corrupt` | a file has the wrong size, or with `--verify` the wrong sha256 | the same command replaces that file only |
| `unverified` | right sizes but no `FETCHED.json`, e.g. copied by hand or with an `rsync` that skipped it | the same command hashes the files and writes it, with no download |
| `stale` | the folder was fetched for a different pin, e.g. after upgrading `mpd` | the same command fetches what changed |
| `unpinned` | `mpd` has no files for this model: AIDO | none, see below |

`mpd submit` stops (exit status 2) unless the teacher and student are `ready` or `override`; `mpd plan` and `mpd submit --dry-run` only
warn. The messages follow one pattern:

```
weights for esm2-3b are missing (/scratch/user/<netid>/mpd/models/esm2-3b). Run on a login node: mpd fetch model esm2-3b --cluster grace
```

When several are not ready, the message lists them and gives one command for all. If a *job* reports this, the folder was deleted or
purged after submission: fetch again and resubmit. (Scratch purge rules are not verified here.)

## ESM2-3B and SaProt need torch 2.7 or newer

`esm2-3b` and both SaProt repositories publish only PyTorch pickle files (`pytorch_model*.bin`); there is no safetensors file on
their `main` branch. Current transformers refuses to unpickle those with torch older than 2.6 (the CVE-2025-32434 hardening). `mpd`
asks for torch 2.7 or newer in any case; `mpd fetch model` prints a reminder after fetching a pickle-only model, and the
preflight check warns if the torch in the planning environment is older than 2.6. The ESM2 models up to 650M are safetensors.

## AIDO is never fetched

`aido-rag-16b` is in `mpd models` only so that `mpd` can say why it is not offered. It has no pinned files, and
`mpd fetch model aido-rag-16b` refuses and quotes the licence note: clause 1.2.2 of the GenBio AI Community License bars using AIDO
outputs to train other models. See [LICENSES.md](LICENSES.md).

## If something goes wrong

| you see | it means | do |
|---|---|---|
| `could not download ...` and a pointer to `--from-dir` | this machine cannot reach huggingface.co | fetch elsewhere and copy, see above |
| `HTTP Error 429` or `503` | Hugging Face is busy or rate-limiting your address | run the command again later; it resumes |
| `not enough free space under ...` | the filesystem has less than the missing bytes plus 1 GiB | pick another `--home` (Grace: `$SCRATCH`), or free space |
| `disk full or quota exceeded` | a write hit your quota | `showquota`, delete old runs, resume |
| `stopped before the login-node time limit` (exit status 3) | the fetch used its time budget | run the same command again |
| `another mpd fetch model ... is already writing` | a fetch of the same model is running | wait for it |
| `sha256 mismatch; removed the file` | the bytes differ from the pin (a bad transfer, or upstream changed) | run again; if it repeats, open an issue with the message |
| `... has no config.json` | `MPD_MODEL_<KEY>` points at the wrong folder | fix or unset the variable |

## What has been tested

The fetch, resume, verification, state and `--from-dir` logic runs in the test suite against a local Hub stand-in with injected
faults (dropped connections, servers that ignore Range, rate limits, corrupted bytes, truncated files, a full disk, a held lock);
all of it on CPU. An opt-in test (`MPD_TEST_NETWORK=1 pytest -m slow`) downloads `esm2-8m` (31 MB) from huggingface.co, resumes it with a
Range request through the CDN, and loads it with the network blocked. The pins were rebuilt from the Hugging Face API on 2026-11-09:
every repository was public, ungated and MIT, the pinned commit was still `main`, and the sizes and sha256 of the large files agreed
between the file listing and the download endpoint. The weights themselves have been hashed from their bytes for `esm2-8m`,
`esm2-650m` and `saprot-650m`; the files of `esm2-35m`, `esm2-150m`, `esm2-3b` and `saprot-35m` rest on that two-endpoint agreement.

Not verified: anything on Grace itself (download speed from its login nodes, the cluster Python's TLS setup, quota behaviour on
`$SCRATCH`, the 60-minute rule in practice), and loading the real SaProt weights.

## For maintainers: changing a pin

`python scripts/maintainers/pin_models.py` rebuilds `models.tsv` at the pinned commits from the Hugging Face API and reports any
difference (exit status 0 means the repository is consistent; `--crosscheck` also compares the large files with a second endpoint).
`--at main --write` rebuilds at each repository's current `main`, writes the manifest and prints the new `revision=` values for
`registry.py`. It needs network, sends no token, and takes about a minute and a half. A new model needs a `ModelInfo` row with a 40-hex
`revision` and a `license`, then the tool: see [ADDING_A_MODEL.md](ADDING_A_MODEL.md).
