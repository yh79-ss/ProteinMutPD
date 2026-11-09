# Security policy

ProteinMutPD is alpha research software that runs on a cluster login node, writes job scripts and submits them with your
account. It downloads data and model weights from public hosts and checks them against pinned checksums.

## Reporting a vulnerability

Report it privately, not in a public issue. Use GitHub's "Report a vulnerability" button on the Security tab of the
repository. *TODO (owner): turn on private vulnerability reporting for the repository; until then, open an issue that says
only "security contact request" and contains no details.*

Examples of what to report: a download that is accepted without its checksum, a path or archive member that escapes the
target directory, a value from the command line or an environment variable that reaches a rendered job script unquoted,
or a message that prints a credential.

## What never goes into an issue, a pull request or a log you share

Slurm account numbers, user names, access tokens, SSH or deploy keys, passwords, and home or project paths that contain
your user name. Replace them with `<ACCOUNT>` and `<netid>`. `mpd doctor --report` is meant to print a summary that is
safe to paste; read it before you post it. If you pasted a secret by mistake, revoke it first and then edit the post.

## Supported versions

There is no tagged release yet. Only the latest commit on `main` is looked at.
