# Security policy

ProteinMutPD is alpha research software that runs on a cluster login node, writes job scripts and submits them with your
account. It downloads data, model weights and, for SaProt, one Foldseek release archive from public hosts and checks each
against a pinned checksum (the VenusMutHub structure deposit, which has no published record yet, is checked against the `SHA256SUMS` of the copy you give it, and a file name in it that leaves the download folder is refused); the Foldseek binary that `mpd fetch foldseek` installs is run only after its own checksum
matches (a binary you name with `MPD_FOLDSEEK` is yours to trust). `scripts/setup_env.sh` installs Python packages from PyPI
at the versions pinned in `constraints/` (versions, not hashes). Published advisories exist for the pinned `transformers`, `torch` and
`accelerate` versions (PyPI and OSV, checked 2026-11-09); several have no fix inside the supported ranges. The jobs are offline, `mpd` loads only
local weight folders whose files match a sha256 manifest, never sets `trust_remote_code`, and calls `torch.load` with `weights_only`. A folder
you name with `MPD_MODEL_<KEY>` or `MPD_MODELS_DIR` is not checked against the manifest, so treat it as trusted input. Dependabot covers
GitHub Actions only; the pins are bumped by hand.

## Reporting a vulnerability

Report it privately, not in a public issue. Use GitHub's "Report a vulnerability" button on the Security tab of the
repository. *TODO (owner): turn on private vulnerability reporting for the repository; until then, open an issue that says
only "security contact request" and contains no details.*

Examples of what to report: a download that is accepted without its checksum, a downloaded program that is run before its
checksum is checked, a path or archive member that escapes the target directory, a value from the command line or an
environment variable that reaches a rendered job script unquoted, or a message that prints a credential.

## What never goes into an issue, a pull request or a log you share

Slurm account numbers, user names, access tokens, SSH or deploy keys, passwords, and home or project paths that contain
your user name. Replace them with `<ACCOUNT>` and `<netid>`. `mpd doctor --report` is meant to print a summary that is
safe to paste; read it before you post it. If you pasted a secret by mistake, revoke it first and then edit the post.

## Supported versions

There is no tagged release yet. Only the latest commit on `main` is looked at.
