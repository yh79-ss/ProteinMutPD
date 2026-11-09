# Cluster profiles

A cluster profile is one YAML file that tells `mpd` what a Slurm site looks like: node size, GPU types and how they are named, partition
and walltime limits, what a service unit costs, where to keep files, and which modules or `#SBATCH` options every job needs. Two
profiles ship with `mpd`:

```text
$ mpd clusters
grace   TAMU HPRC Grace (Slurm, 48-core nodes). GPU nodes are A100 40 GB / A40 48 GB / RTX 6000 24 GB / T4 16 GB.
         home $SCRATCH/mpd, login node: 8 cores / 60 min
   a100  2/node  40 GB  partition=gpu
   a40   2/node  48 GB  partition=gpu-a40
   rtx   2/node  24 GB  partition=gpu  no native bf16: training is refused (--allow-unsupported-gpu)
   t4    4/node  16 GB  partition=gpu  no native bf16: training is refused (--allow-unsupported-gpu)
vision  TAMU Vision cluster (8 x H200 per node). Development profile.
         home $MPD_HOME, login node: 8 cores / 60 min
   h200  8/node  141 GB  partition=def
```

For another site you write your own and pass its path: `mpd plan --cluster ./mycluster.yaml ...` (anything ending in `.yaml` or `.yml`,
or containing a `/`, is read as a file; `MPD_CLUSTER=./mycluster.yaml` does the same). `mpd clusters` lists only the packaged profiles.
This page is the field reference, a commented template, and how to check a profile before you trust it.

Tags as in [GRACE.md](GRACE.md): **[CPU]** tested on CPU here, **[Vision]** accepted by `sbatch --test-only` on the Vision cluster,
**[NOT VERIFIED]** nobody has run it.

## How a profile is checked

Every statement in this section is tested on CPU (`tests/test_cluster.py`).

- An unknown field is an error that lists the valid ones (`gpus.a100: unknown field speed; valid fields are ...`), so a typo such as
  `sbatch_extras` is never silently ignored. A missing or invalid field is an error that names the file and the field:
  `my.yaml: gpus.a100.per_node: expected a whole number, got 'two'`.
- Walltimes must be **quoted** strings: YAML reads an unquoted `48:00:00` as the number 172800, and `mpd` says so.
- `sbatch_extra` entries must be single long options (`--name` or `--name=value`, no spaces), and `modules` entries module names
  (letters, digits, `_ . + / -`), because both end up verbatim in job scripts.
- `account` may be written as a number (`account: 123456789012`); `mpd` reads it as text.
- Never put a real account number in a profile you share. Pass it with `--account` or `MPD_ACCOUNT`.

## Top level

| field | required | default | meaning |
|---|---|---|---|
| `name` | yes | | Letters, digits, `_ . -`. It appears in messages and in the campaign id hash, and `scripts/setup_env.sh` records the **file name** of the profile in `<home>/env/mpd_env.json`; the planner ignores a record whose cluster differs from `name`. So name the file `<name>.yaml` (see [below](#using-a-profile-with-the-setup-script)). |
| `description` | no | `""` | One line shown by `mpd clusters`. |
| `account` | no | `null` | Billing account written as `#SBATCH --account=` when neither `--account` nor `$MPD_ACCOUNT` is given. Order: `--account`, `MPD_ACCOUNT`, this field, Slurm's default. `[A-Za-z0-9_.-]+`, no length rule. |
| `node` | yes | | The compute node, below. |
| `sbatch_extra` | no | `[]` | Options added as `#SBATCH` lines to every script, after `mpd`'s own. For something every job on the site needs (`--qos=standard`). **Not** `--export` or `--get-user-env`: Slurm 25.05 rejects `--get-user-env=L` and `--export=NONE` implies `--get-user-env` **[Vision]**; `mpd` cleans the inherited environment itself. |
| `modules` | no | `[]` | Modules every job loads. The modules recorded by `scripts/setup_env.sh` and `--module` are added after these (order kept, repeats dropped), so a student normally leaves this empty. |
| `login_node` | no | `max_cores: 8`, `max_minutes: 60` | Login-node limits; `mpd fetch` stops itself 5 minutes before `max_minutes`. |
| `gpus` | yes | | Mapping of GPU key to its description (below); at least one. The key is what `--gpu` takes. |
| `default_gpu` | yes | | A key of `gpus`, used when `--gpu` is absent. |
| `default_train_gpus` | no | `null` | Width of `*-all` training when `--train-gpus` is absent. `null`: the widest of up to 4 GPUs whose count times the micro-batch divides the global batch. A number: that many (fewer if the global batch does not divide). Grace sets 1 because DDP has never run on a real GPU and a two-GPU request takes a whole node. |
| `array_throttle` | no | `4` | Concurrent array tasks (`%N` in `--array`); `--array-throttle` overrides. At least 1. |
| `resource_rule` | no | `proportional_to_node` | How cores and memory follow the GPUs: `proportional_to_node` gives the GPUs' share of the node (HPRC's `maxconfig` shape: 24 cores and 180 GB per A100 on Grace); `fixed_per_gpu` uses the numbers in `fixed_per_gpu`. |
| `fixed_per_gpu` | with that rule | | `cpus` and `mem_gb` per GPU, for example `{cpus: 8, mem_gb: 64}` (Vision). |
| `cpu` | no | below | CPU-only stages (`prepare`, `evaluate`). |
| `storage` | no | `root: "{env:MPD_HOME}"` | Where the working root lives. |

`node`:

| field | required | default | meaning |
|---|---|---|---|
| `cores` | yes | | Cores per compute node. |
| `usable_mem_gb` | yes | | Memory a job may ask for per node, in GB (Grace: 360 of 384). The proportional rule uses it. |
| `su_mem_total_gb` | no | `usable_mem_gb` | Memory that bills a whole node in the service-unit rule `effective cores = max(cores, min(node cores, ceil(mem / su_mem_total_gb * node cores)))`. Grace: 351.5625, from HPRC's AMS page; with it `maxconfig`'s 25 effective cores for 24 cores and 180 GB come out. |

`gpus.<key>`:

| field | required | default | meaning |
|---|---|---|---|
| `per_node` | yes | | GPUs of this type in one node. `--train-gpus` above it is an error (`does not fit one node`); multi-node jobs are never generated. |
| `mem_gb` | yes | | GPU memory in GB. The planner keeps 15% free and chooses the micro-batch from it. |
| `max_walltime` | yes | | Longest request the partition accepts, as a **quoted** string. A longer request is an error, because Slurm would hold the job forever as `PartitionTimeLimit`. |
| `partition` | no | `null` | `#SBATCH --partition=`; `null` lets Slurm choose. |
| `gres` | no | the key | The type in `--gres=gpu:<type>:N`, spelled as the site spells it. Vision needs `H200`; lower-case `h200` is rejected **[Vision]**. See `sinfo -o %G`. |
| `su_per_gpu_hour` | no | `0` | Service units per GPU-hour. `0` means "not billed per SU": the plan then prints `not billed`. |
| `time_factor` | no | `1.0` | Planning slowdown against one H200, the GPU the timings were taken on. An assumption unless you measured it; walltimes that depend on it are labelled estimates. |
| `max_nodes` | no | `null` | Informational. |
| `bf16` | no | `true` | `false` for GPUs without native bfloat16 (compute capability below 8.0: Turing T4 and RTX 6000). `mpd` then refuses to train on them, naming alternatives, unless `--allow-unsupported-gpu` (float32). |
| `unverified` | no | `null` | A sentence `mpd` prints as a warning whenever this GPU is used. Use it for facts nobody confirmed. |

`cpu`:

| field | default | meaning |
|---|---|---|
| `partition` | `null` | Partition for CPU stages. Grace leaves it out: it routes by walltime, and an explicit `short` with more than 2 hours waits forever. |
| `max_walltime` | `"1-00:00:00"` | Longest CPU job, quoted. |
| `su_per_core_hour` | `0` | SU per effective core-hour. |
| `gres` | `null` | Written as `#SBATCH --gres=` on CPU stages only, for sites that reject jobs without a GPU request. Vision prints "Jobs without a GPU request will soon be rejected" for them; `gres: "gpu:1"` silences it **[Vision]**, at the price of reserving a GPU for the short CPU stages. Form `gpu:N` or `gpu:<type>:N`. |

`storage`:

| field | default | meaning |
|---|---|---|
| `root` | `"{env:MPD_HOME}"` | Default working root. `{env:VAR}` is replaced by the environment variable when `mpd` runs; an unset or empty variable is an error that says how to pass `--home`. The result is always made absolute, because a job changes directory. |
| `forbidden_prefixes` | `[]` | Absolute paths the root may not be under (symlinks are followed). Grace: `["/home"]`. |
| `home_max_files`, `home_max_gb` | `null` | Limits of the forbidden area; used only in the error text. |
| `file_quota_warning` | `null` | File count of the working area above which `mpd doctor` may warn. |

## A complete template

Copy it, change the numbers to what `sinfo`, `scontrol show node` and your site's documentation say, and save it as
`<name>.yaml`. It is loaded and planned in this repository's check of this page **[CPU]**.

```yaml
# mycluster.yaml - a made-up site with 32-core nodes and two GPU types.
name: mycluster                  # the file must be called mycluster.yaml (see "Using a profile with the setup script")
description: Example site with 32-core nodes and two GPU types.

account: null                    # leave null in anything you share; pass --account or export MPD_ACCOUNT

node:
  cores: 32
  usable_mem_gb: 250             # what --mem may ask for per node
  # su_mem_total_gb: 250         # only if the site's SU rule uses another total than usable_mem_gb

sbatch_extra: []                 # for example ["--qos=normal"]; never --export or --get-user-env
modules: []                      # setup_env.sh records the modules the environment needs; list one here only if every job needs it

login_node:
  max_cores: 8
  max_minutes: 60

gpus:
  a100:                          # the key is what --gpu takes
    per_node: 4
    mem_gb: 80
    max_walltime: "2-00:00:00"   # quoted
    partition: gpu
    gres: A100                   # as `sinfo -o %G` spells it
    su_per_gpu_hour: 0           # 0: not billed per SU
    time_factor: 3.0             # assumption: slowdown against one H200
    max_nodes: 8
  t4:
    per_node: 4
    mem_gb: 16
    max_walltime: "1-00:00:00"
    partition: gpu
    bf16: false                  # Turing: training is refused unless --allow-unsupported-gpu
    time_factor: 15.0
    unverified: "t4 limits were copied from the site's web page and not tried."   # printed as a warning
default_gpu: a100
default_train_gpus: 1

cpu:
  partition: null                # null: let Slurm route CPU stages by walltime
  max_walltime: "2-00:00:00"
  su_per_core_hour: 0
  # gres: "gpu:1"                # only if the site rejects jobs without a GPU request

resource_rule: proportional_to_node   # or: fixed_per_gpu, with a fixed_per_gpu: {cpus: 8, mem_gb: 64} block
array_throttle: 4

storage:
  root: "{env:SCRATCH}/mpd"
  forbidden_prefixes: ["/home"]
  home_max_files: 10000
  home_max_gb: 10
  file_quota_warning: 250000
```

## Checking a new profile

Each step needs no job and no GPU.

1. `mpd plan --cluster ./mycluster.yaml --home /tmp/mpd-test --teacher esm2-8m --student esm2-8m --train-data proteingym --method score-all --eval-data proteingym`
   loads the profile (a mistake stops with the field name) and prints the stage table. Compare its GPU type, partition, cores, memory and
   walltimes with what `sinfo` and your site's documentation say. The `[FAIL]` lines about a missing environment or data are about your
   machine, not the profile.
2. `mpd submit ... --dry-run` with the same flags writes the scripts under `<home>/runs/<campaign>/scripts/` without queueing anything.
   Read one, and run `bash -n` on each.
3. On the cluster, `sbatch --test-only <script>` for one script of each shape: a CPU stage, a one-GPU stage, a DDP stage (`--train-gpus 2`
   or more) and an array stage. It queues nothing. (`mpd submit --dry-run` already does this for every distinct shape when `sbatch` is on
   the `PATH`.) This is how the case-sensitive GRES type of Vision was found **[Vision]**.
4. `scontrol show config | grep MaxArraySize`: `mpd` splits arrays at 1,000 tasks, which is safe if the limit is at least 1,001.
5. Run the [cheap first run](GRACE.md#your-first-run-cheap) of your site's equivalent before a real campaign.

## Using a profile with the setup script

`bash scripts/setup_env.sh --cluster ./mycluster.yaml --home /path/to/home --modules "<modules>"` builds the environment and records the
**file name without extension** (`mycluster`) as the cluster. If the profile's `name:` differs, the planner ignores the record and warns
`it was written for cluster 'mycluster', not 'othername', so the planner ignores it`, and the jobs would then not load the recorded modules
**[CPU]**. Keep the two the same. The script has Grace defaults (module, home) only for `--cluster grace`; for any other profile give `--home`, and `--modules` if
your Python comes from a module.

## What Vision verified

The Vision profile (8 x H200 per node, partition `def`, `--qos=standard`, Slurm 25.05.6) is the development cluster. With `sbatch
--test-only`, nothing queued **[Vision]**:

| tried | result |
|---|---|
| `--gres=gpu:H200:1` | accepted |
| `--gres=gpu:h200:1` | rejected: "Requested node configuration is not available" |
| `#SBATCH --get-user-env=L` | rejected: "option '--get-user-env' doesn't allow an argument" |
| `--export=NONE` alone, or a bare `--get-user-env` | accepted |
| 50 rendered scripts: 4 methods x the six stages, one-GPU and four-GPU training, a 1,001-task array with its single-task last chunk, `--sbatch-arg` and `--env` variants | all accepted |
| CPU stages | accepted, with Vision's warning "this job requests no GPU. Jobs without a GPU request will soon be rejected"; `cpu: gres: "gpu:1"` removes it |
| `scontrol show config` | `MaxArraySize = 1001`, `MaxJobCount = 10000` |

Everything else in the Vision profile came from working Slurm scripts there and from `sinfo` and `sacctmgr`. The Grace profile has not
been tried on Grace; its header lists what is not verified, and [GRACE.md](GRACE.md#what-has-not-been-verified-on-grace) explains it.

## Contributing a profile

Open a pull request with `src/mpd/clusters/<name>.yaml` (packaged files are listed by `mpd clusters`), with the evidence for each number in
comments the way `grace.yaml` does (`doc`, `ran` or `unverified`), no account number, user name or project name, and the output of
step 3 above. See [CONTRIBUTING.md](../CONTRIBUTING.md).
