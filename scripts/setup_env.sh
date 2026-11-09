#!/usr/bin/env bash
# Build the Python environment mpd's jobs run in: a venv under <home>/env, pip-installed from this clone and pinned
# by constraints/grace.txt. Run it on a login node (it needs the internet). It writes only under <home>; $HOME is read, never written.
#
#   bash scripts/setup_env.sh --cluster grace
#
# Status: NOT RUN ON GRACE. The Grace defaults below (module Anaconda3/2024.02-1, home $SCRATCH/mpd) come from earlier
# jobs and HPRC documents, not from a run of this script there. Everything else is tested with fake tools (tests/test_setup_script.py).

if [ -z "${BASH_VERSION:-}" ]; then
  echo "setup: error: run this script with bash:  bash scripts/setup_env.sh --cluster grace" >&2
  exit 2
fi
# No `set -u` yet: /etc/bashrc and the module scripts on Grace fail under it. It is switched on after the modules are loaded.
set -eo pipefail

SELF=${BASH_SOURCE[0]:-$0}
REPO=$(cd "$(dirname "$SELF")/.." && pwd)
dry=0
[ "${MPD_SETUP_DRY_RUN:-}" = 1 ] && dry=1

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_env.sh [--cluster grace|vision|NAME] [--home DIR] [--python PATH]
                                 [--modules "M1 M2"|none] [--dev] [--no-torch] [--constraints FILE|none]
                                 [--rebuild] [--yes]

Builds <home>/env (a venv), installs mpd from this clone with pip, records how in <home>/env/mpd_env.json (the job
scripts load the same modules), adds MPD_HOME and MPD_CLUSTER to <home>/env/bin/activate, and finishes with `mpd doctor`.
Run it on a login node. It is safe to run again: it reuses the environment, updates the install and rewrites the record.

  --cluster C        profile name or YAML path (required; default $MPD_CLUSTER)
  --home DIR         working root (default $MPD_HOME, else $SCRATCH/mpd on grace). On grace a home under /home is refused:
                     the quota is 10 GB and 10,000 files, and an environment has tens of thousands of files
  --python PATH      interpreter to build the venv from (default: python3 after the modules are loaded; 3.11 or newer)
  --modules "M1 M2"  modules to load first, or "none" (default $MPD_SETUP_MODULES, else Anaconda3/2024.02-1 on grace, else none)
  --constraints F    pip constraints (default constraints/grace.txt of this clone: the tested torch/transformers/peft set);
                     "none" installs the loose ranges of pyproject.toml, which is NOT the tested set
  --dev              editable install with the test extra, for working on mpd itself. The default is a normal install, built from
                     a copy of the sources in <home>/cache/install-src, so a `git pull` cannot change the code under a running
                     campaign and pip writes nothing into the clone: run this script again to update
  --no-torch         install only the planner (no torch): enough to plan and submit from a machine that runs no job
  --rebuild          delete <home>/env first (only a folder that holds a pyvenv.cfg) and build it again
  --yes              do not ask for confirmation

Environment: MPD_SETUP_DRY_RUN=1 prints the commands instead of running the ones that write. pip reads PIP_INDEX_URL and friends.
Login nodes end processes after 60 minutes (HPRC primer). If that happens run the script again: downloaded wheels are cached
in <home>/cache/pip, so it continues where it was.
Not run on Grace: the Grace defaults come from earlier jobs and HPRC documents, not from a run of this script there.
Exit status: 0 done, 1 a step failed, 2 refused (bad option, home under /home, no confirmation).
EOF
}

say() { printf 'setup: %s\n' "$*"; }
warn() { printf 'setup: warning: %s\n' "$*" >&2; }
fail() { printf 'setup: error: %s\n' "$*" >&2; exit 1; }
refuse() { printf 'setup: error: %s\n' "$*" >&2; exit 2; }
run() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  [ "$dry" = 1 ] || "$@"
}

# ---------------------------------------------------------------- options
cluster=${MPD_CLUSTER:-}
home=
python=
modules=
modules_given=0
constraints=
dev=0
no_torch=0
rebuild=0
yes=0

want_value() { [ "$2" -ge 2 ] || refuse "$1 needs a value (bash scripts/setup_env.sh --help)"; }
while [ $# -gt 0 ]; do
  case $1 in
    --cluster|--home|--python|--modules|--constraints) want_value "$1" "$#"; opt=${1#--}; val=$2; shift 2 ;;
    --cluster=*|--home=*|--python=*|--modules=*|--constraints=*) opt=${1%%=*}; opt=${opt#--}; val=${1#*=}; shift ;;
    --dev) dev=1; shift; continue ;;
    --no-torch) no_torch=1; shift; continue ;;
    --rebuild) rebuild=1; shift; continue ;;
    --yes|-y) yes=1; shift; continue ;;
    -h|--help) usage; exit 0 ;;
    *) refuse "unknown option $1 (bash scripts/setup_env.sh --help)" ;;
  esac
  case $opt in
    cluster) cluster=$val ;;
    home) home=$val ;;
    python) python=$val ;;
    modules) modules=$val; modules_given=1 ;;
    constraints) constraints=$val ;;
  esac
done

abspath() {
  # shellcheck disable=SC2088  # "~/" is matched literally on purpose: a tilde typed by the user that the shell did not expand
  case $1 in
    "~") printf '%s\n' "${HOME:-/}" ;;
    "~/"*) printf '%s\n' "${HOME:-}/${1#"~/"}" ;;
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$PWD/$1" ;;
  esac
}
normalize() {                 # collapse //, /./ and /../ without touching the file system
  local IFS=/ part out=""
  set -f
  for part in $1; do
    case $part in
      ""|.) ;;
      ..) out=${out%/*} ;;
      *) out=$out/$part ;;
    esac
  done
  set +f
  printf '%s\n' "${out:-/}"
}

# ---------------------------------------------------------------- cluster defaults, home, modules, constraints
[ -n "$cluster" ] || refuse "give --cluster NAME (grace, vision, or your own profile) or set MPD_CLUSTER: the environment record names it"
case $cluster in            # Grace and GRACE mean grace, as they do for mpd; a path to a profile file keeps its spelling
  */*|*.yaml|*.yml) ;;
  *) cluster=$(printf '%s' "$cluster" | LC_ALL=C tr '[:upper:]' '[:lower:]') ;;
esac
default_modules=
default_home=
no_home_under=
if [ "$cluster" = grace ]; then
  default_modules=Anaconda3/2024.02-1
  no_home_under=/home
  if [ -n "${SCRATCH:-}" ]; then default_home=$SCRATCH/mpd; fi
fi
[ -n "$home" ] || home=${MPD_HOME:-$default_home}
if [ -z "$home" ]; then
  if [ "$cluster" = grace ]; then
    refuse "\$SCRATCH is not set, so there is no default home. Pass --home DIR (on scratch, not under /home)"
  fi
  refuse "no home: pass --home DIR or set MPD_HOME (the folder for the environment, data, models and runs)"
fi
home=$(normalize "$(abspath "$home")")
if [ -n "$no_home_under" ]; then
  case $home in
    "$no_home_under"|"$no_home_under"/*)
      refuse "$home is under $no_home_under (10 GB / 10,000 files on Grace). Use scratch: export MPD_HOME=\$SCRATCH/mpd" ;;
  esac
fi
case $home in
  "${HOME:-/nonexistent}"|"${HOME:-/nonexistent}"/*) warn "$home is inside your home directory; check that its quota is large enough" ;;
esac

[ "$modules_given" = 1 ] || modules=${MPD_SETUP_MODULES:-$default_modules}
[ "$modules" = none ] && modules=

if [ "$constraints" = none ]; then
  constraints=
  warn "no constraints: pip resolves the loose ranges in pyproject.toml, which is not the tested torch/transformers/peft set"
else
  [ -n "$constraints" ] || constraints=$REPO/constraints/grace.txt
  constraints=$(normalize "$(abspath "$constraints")")
  [ -f "$constraints" ] || refuse "constraints file $constraints not found. It pins the tested package set; pass --constraints FILE, or --constraints none for the loose ranges of pyproject.toml (not the tested set)"
fi
[ -f "$REPO/pyproject.toml" ] || fail "$REPO has no pyproject.toml: run this script from a clone of the repository"

env_dir=$home/env
cache=$home/cache
record_cluster=$(basename "$cluster"); record_cluster=${record_cluster%.yaml}; record_cluster=${record_cluster%.yml}

say "cluster $cluster, home $home"
say "modules: ${modules:-none}; constraints: ${constraints:-none}"

# ---------------------------------------------------------------- modules and the interpreter
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

loaded() { case ":${LOADEDMODULES:-}:" in *":$1:"*|*":$1/"*) return 0 ;; esac; return 1; }
if [ -n "$modules" ]; then
  profile=${MPD_SETUP_PROFILE:-/etc/profile}       # where a login shell gets `module` from; the variable is for tests
  # shellcheck source=/dev/null
  type module >/dev/null 2>&1 || { if [ -r "$profile" ]; then . "$profile" >/dev/null 2>&1 || true; fi; }
  type module >/dev/null 2>&1 || fail "the 'module' command is not available in this shell. Use a login shell, or give a Python directly: --modules none --python /path/to/python3"
  module purge || fail "module purge failed"
  set -f
  # shellcheck disable=SC2086
  module load $modules || true            # a missing module is judged below; some module systems still return 0
  for m in $modules; do
    loaded "$m" || fail "module $m did not load. On Grace look for a Python module with:  module spider Python  (or: module spider Anaconda3)
  then pass it:  bash scripts/setup_env.sh --cluster $cluster --modules \"<module names>\"
  or use a Python you already have:  --modules none --python /path/to/python3
  Use Python 3.11, 3.12 or 3.13: newer versions were not tested.
  (the Grace default is the module earlier jobs loaded; this script has not been run on Grace)"
  done
  set +f
fi
unset PYTHONPATH PYTHONHOME               # a module may have set them
set -u

if [ -z "$python" ]; then
  python=$(command -v python3 || command -v python || true)
  [ -n "$python" ] || fail "no python3 on PATH${modules:+ after loading $modules}. Load a Python module (module spider Python) or pass --python"
else
  case $python in */*) python=$(normalize "$(abspath "$python")") ;; esac       # a bare name is looked up on PATH
fi
py_version=$("$python" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))') || fail "$python cannot be run"
case $py_version in
  [0-2].*|3.[0-9].*|3.10.*) fail "$python is Python $py_version; mpd needs 3.11 or newer. Load a newer Python module (module spider Python) or pass --python" ;;
esac
py_minor=${py_version#*.}; py_minor=${py_minor%%.*}
if [ "${py_version%%.*}" -gt 3 ] || [ "$py_minor" -gt 13 ]; then       # constraints/grace.txt: tested on 3.11, 3.12 and 3.13
  warn "$python is Python $py_version, which is newer than the Python 3.13 that mpd was tested with. It may work. If the install or a job fails, build again with Python 3.11, 3.12 or 3.13 (--modules or --python)"
fi
"$python" -c 'import venv, ensurepip' >/dev/null 2>&1 || fail "$python has no venv or ensurepip module (some distributions package them separately). Use another Python"
say "python $py_version ($python)"

# ---------------------------------------------------------------- what will be installed
extras=
[ "$no_torch" = 1 ] || extras=train
[ "$dev" = 0 ] || extras=${extras:+$extras,}test
# A normal install builds from a copy of the sources: pip then writes no build/ or *.egg-info into the clone (it may be
# read-only), and a build/lib left by an earlier run cannot put files that a `git pull` deleted back into the package.
if [ "$dev" = 1 ]; then source_dir=$REPO; else source_dir=$cache/install-src; fi
target=$source_dir${extras:+[$extras]}
pip_install=(install)
[ -z "$constraints" ] || pip_install+=(-c "$constraints")
if [ "$dev" = 1 ]; then pip_install+=(-e "$target"); else pip_install+=("$target"); fi

stage_sources() {
  { rm -rf "$source_dir" && mkdir -p "$source_dir"; } || fail "cannot write $source_dir"
  local item
  for item in "$REPO"/*; do
    if [ -d "$item" ]; then
      if [ "$(basename "$item")" = src ]; then cp -R "$item" "$source_dir/"; fi
    else
      cp "$item" "$source_dir/"
    fi
  done
  find "$source_dir" \( -name '*.egg-info' -o -name __pycache__ \) -prune -exec rm -rf {} + 2>/dev/null || true
}

reuse=0
if [ -e "$env_dir" ]; then
  if [ "$rebuild" = 1 ]; then
    [ -f "$env_dir/pyvenv.cfg" ] || refuse "$env_dir exists but is not a venv (no pyvenv.cfg), so --rebuild will not delete it. Move it away first"
  elif [ -f "$env_dir/pyvenv.cfg" ] && [ -x "$env_dir/bin/python" ]; then
    reuse=1
  elif [ -n "$(ls -A "$env_dir" 2>/dev/null)" ]; then
    refuse "$env_dir exists but is not a working venv. Move it away, or use --rebuild if it is a broken venv"
  fi
fi

if [ "$dry" != 1 ] && [ "$yes" != 1 ]; then
  [ -t 0 ] || refuse "no terminal to ask on: pass --yes to go ahead"
  say "about to $([ "$reuse" = 1 ] && echo update || echo build) $env_dir with pip: roughly 3 GB to download from PyPI and tens of thousands of files"
  say "(a size from earlier environments, not measured by this script). Packages are cached in $cache/pip, not in your home directory."
  printf 'Continue? [y/N] '
  read -r answer
  case $answer in y|Y|yes|YES) ;; *) refuse "not confirmed" ;; esac
fi

# ---------------------------------------------------------------- build
run mkdir -p "$cache/pip" "$cache/tmp"
export PIP_CACHE_DIR=$cache/pip TMPDIR=$cache/tmp PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
if [ "$rebuild" = 1 ] && [ -e "$env_dir" ]; then
  say "removing the old environment $env_dir"
  run rm -rf "$env_dir"
  reuse=0
fi
if [ "$reuse" = 1 ]; then
  say "reusing the environment $env_dir"
else
  run "$python" -m venv "$env_dir"
fi
venv_python=$env_dir/bin/python
if [ "$dev" != 1 ]; then
  if [ "$dry" = 1 ]; then say "would copy the sources of $REPO (src/ and the top-level files) to $source_dir"; else stage_sources; fi
fi
say "installing ${target} (needs the internet: run this on a login node)"
run "$venv_python" -m pip "${pip_install[@]}"

if [ "$dry" != 1 ]; then
  files=$(find "$env_dir" -type f | wc -l | tr -d ' ')
  kb=$(du -sk "$env_dir" | cut -f1)
  say "environment: $files files, $((kb / 1024)) MB (Grace quotas: home 10,000 files, scratch 250,000 files; check with showquota)"
  [ "$cluster" != grace ] || [ "$files" -lt 100000 ] || warn "$files files is more than 40% of the 250,000-file scratch quota"
fi

# ---------------------------------------------------------------- record and activation
if [ "$dry" = 1 ] || "$venv_python" -c 'import mpd.envrecord' >/dev/null 2>&1; then
  run "$venv_python" -m mpd.envrecord write --home "$home" --cluster "$record_cluster" --modules "${modules:-none}" ${constraints:+--constraints "$constraints"}
else
  warn "this mpd has no mpd.envrecord module; writing the same schema-1 record from the script"
  "$venv_python" - "$home" "$record_cluster" "${modules:-none}" "${constraints:-}" <<'EOF'
import hashlib, json, os, platform, sys
from datetime import datetime, timezone
from importlib import metadata
home, cluster, modules, constraints = sys.argv[1:5]
packages = {}
for name in ("torch", "transformers", "peft", "numpy", "scipy", "pyyaml"):
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
try:
    mpd_version = metadata.version("proteinmutpd")
except metadata.PackageNotFoundError:
    mpd_version = "unknown"
record = {"schema": 1, "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "cluster": cluster,
          "home": home, "python": sys.executable, "python_version": platform.python_version(),
          "modules": [] if modules == "none" else modules.split(), "mpd_version": mpd_version,
          "constraints": constraints or None,
          "constraints_sha256": hashlib.sha256(open(constraints, "rb").read()).hexdigest() if constraints else None,
          "packages": packages}
path = os.path.join(home, "env", "mpd_env.json")
with open(path + ".tmp", "w") as fh:
    fh.write(json.dumps(record, indent=2) + "\n")
os.replace(path + ".tmp", path)
print("wrote", path)
EOF
fi

activate=$env_dir/bin/activate
block_begin='# >>> mpd (written by scripts/setup_env.sh; running it again replaces this block) >>>'
block_end='# <<< mpd <<<'
if [ "$dry" = 1 ]; then
  say "would add to $activate: export MPD_HOME=$home and export MPD_CLUSTER=$cluster"
else
  [ -f "$activate" ] || fail "$activate does not exist: the venv is incomplete (try --rebuild)"
  tmp=$activate.mpd-tmp
  awk -v b="$block_begin" -v e="$block_end" '$0 == b {skip = 1} !skip {print} $0 == e {skip = 0}' "$activate" > "$tmp"
  {
    printf '%s\n' "$block_begin"
    printf 'export MPD_HOME=%q\n' "$home"
    printf 'export MPD_CLUSTER=%q\n' "$cluster"
    printf '%s\n' "$block_end"
  } >> "$tmp"
  cat "$tmp" > "$activate"
  rm -f "$tmp"
  say "activation: source $activate   (sets MPD_HOME and MPD_CLUSTER)"
fi

# ---------------------------------------------------------------- finish with the doctor
say "checking the result with mpd doctor"
rc=0
run "$venv_python" -m mpd doctor --cluster "$cluster" --home "$home" || rc=$?
if [ "$dry" = 1 ]; then
  say "dry run: nothing was written"
  exit 0
fi
say "next: source $activate, then fetch what a run needs (login node):"
say "  mpd fetch proteingym ; mpd fetch venusmuthub ; mpd fetch model esm2-3b esm2-650m"
if [ "$rc" -ne 0 ]; then
  [ "$no_torch" = 1 ] && { say "mpd doctor exited with status $rc. With --no-torch a failing 'packages' row is expected (no torch by design); read the output above for anything else"; exit 0; }
  fail "mpd doctor reported problems (above): fix them, or run it again with --rebuild"
fi
say "done"
