# Licences and permitted use

A plain-language summary of which licence covers which part of ProteinMutPD, and what `mpd` will not do because of them. **It is not
legal advice and it is not a permission from anyone.** The licence texts are the authority; where this page and a licence text differ, the
licence text wins. The statements were checked against the upstream sources on 2026-11-09 (how: see the last section).

## This repository

The ProteinMutPD code is MIT-licensed ([LICENSE](../LICENSE)). The repository contains no model weights, no benchmark files and no
cached model outputs. `mpd fetch` and the stage code download or load third-party files at run time; those files keep the licences
below, and you accept them when you download them.

## Everything `mpd` touches

| material | role | licence (as stated upstream) | in `mpd` |
|---|---|---|---|
| ProteinGym v1.3, assays and AF2 structures ([Zenodo 10.5281/zenodo.15293562](https://doi.org/10.5281/zenodo.15293562)) | benchmark and training sequences | MIT (Zenodo record metadata `mit-license`) | `mpd fetch proteingym` |
| VenusMutHub (`AI4Protein/VenusMutHub`, revision `08a27d07`) | benchmark and training sequences | MIT (dataset card; Hugging Face licence tag `mit`) | `mpd fetch venusmuthub` |
| Venus structure set (Zenodo deposit **in preparation**, [ZENODO.md](ZENODO.md)) | structures for SaProt | CC BY 4.0 for the deposit as a whole; it contains wwPDB entries (CC0 1.0), AlphaFold DB models (CC BY 4.0) and ESMFold predictions (`facebook/esmfold_v1`, MIT); attribution is in the deposit's README | `mpd fetch venus-structures`; not published yet |
| ESM2 weights, 8M to 3B (`facebook/esm2_*`) | teacher and student | MIT (model cards) | available |
| SaProt weights, 35M and 650M (`westlake-repl/SaProt_*_AF2`) | teacher and student | MIT (both model cards and the GitHub repository) | `planned`: fetchable, not yet connected to the stages ([SAPROT.md](SAPROT.md)) |
| Foldseek | computes the 3Di structure alphabet SaProt reads | GPL-3.0 | downloaded from its owners by `mpd fetch foldseek`, run as a separate program, **never bundled** (below) |
| biotite (optional, `MPD_3DI_ENGINE=biotite`) | pure-Python 3Di fallback | BSD-3-Clause | installed by you with pip |
| AIDO.Protein-RAG-16B (`genbio-ai/GB.Protein-RAG-16B-proteingym-dms-zeroshot`) | teacher in the earlier study | GenBio AI Community License (`license: other` on Hugging Face) | **not included**; blocked as a teacher of non-AIDO students (below) |
| Python packages: PyTorch BSD-3-Clause, Transformers and PEFT Apache-2.0, NumPy and SciPy BSD, PyYAML MIT | | from the packages' metadata | installed by pip, not redistributed here |

`mpd models` shows the licence of each model, and `mpd fetch model --list` the pinned commit of the weights.

### Data caveats

ProteinGym and VenusMutHub gather measurements from many original papers. Cite the benchmark you use and, where you can, the original
sources: VenusMutHub's dataset card asks you to cite it, and its `doi.csv` maps every assay to the original publication; ProteinGym's
reference table (`DMS_substitutions.csv`, fetched by `mpd fetch proteingym`) names the source of each assay.

## Foldseek (GPL-3.0)

`mpd` downloads the `foldseek` program from its owners onto your machine and runs it as a separate process: it passes a structure file on the
command line and reads a text file back. The GNU GPL FAQ says that pipes, sockets and command-line arguments are normally communication
between separate programs, unless the exchange is intimate enough to share complex internal data structures
([FAQ, "aggregate"](https://www.gnu.org/licenses/gpl-faq.html#MereAggregation)); a file name in and a text file out is not that.
Obligations such as a source offer and the licence text arise if **you redistribute the binary**, so this repository, its wheels,
container images and the Zenodo deposit never include it. If you put a Foldseek binary into anything you share, keep its licence with it.
This is not legal advice.

## AIDO and the GenBio AI Community License

### What `mpd` does

- AIDO is **not available** in this release. `mpd` cannot plan, download or run it, `mpd fetch` has no AIDO source, and the repository
  contains no AIDO weights, code, structure embeddings, target caches or adapters.
- `mpd` refuses any run in which AIDO is the **teacher** of a model that is not an AIDO model, whatever the student is, with the clause
  quoted below. The check is in `RunSpec.validate` (through `output_family_lock`), it runs before the "planned" message, and it stays in
  force if AIDO support is added later **[CPU]** (`tests/test_licenses.py`). There is no switch to override it: that would need
  written permission from GenBio and a change to the code.
- The check is deliberately stricter than the licence's wording ("similar" models): deciding what is similar is GenBio's to say, not a
  script's, so every non-AIDO student is refused.
- AIDO is listed in `mpd models` only so that the planner can explain this. As the earlier study's AIDO-teacher results cannot be
  produced under this rule, they cannot be reproduced with `mpd` ([RESULTS.md](RESULTS.md#comparing-with-the-earlier-study)).

### What the licence says

The licence file of the model repository is 10,686 bytes with SHA-256
`fd27fba1a72bd58054dd59dbbf2b4f4c5e570a1115536828e1fd8ca8496bc861` (the file at `raw/main/LICENSE` was fetched and hashed on 2026-11-09 and
matched this value). Clause 1.2.2, verbatim:

> 1.2.2 You shall not, directly or indirectly: (a) use or provide access to any Outputs or Output Derivatives to train, optimize, improve,
> or otherwise enhance the functionality or performance of any machine learning models or related technologies that are similar to the
> GenBio Materials; (b) engage in any form of model distillation or other methods that would achieve the purposes described in
> subsection (a) above. Notwithstanding the foregoing, you may use Outputs and Output Derivatives to train, optimize, improve, or
> enhance the functionality or performance of: (i) The GenBio Materials itself; and (ii) downstream Derivative Works of the GenBio Materials;

"Output" (clause 8.7) is any output, and expressly includes "any intermediate results, such as activations across model layers,
intermediate outputs from model layers (e.g., attention maps), as well as gradients and embeddings produced by the GenBio Materials".
Logits, scores and any table made from them are therefore Outputs, as the text reads. Use is limited to non-commercial purposes (clauses
1.1 and 1.2.1), and "the provision of Outputs as a service is not a Non-Commercial Purpose" (clause 8.5). Clause 1.2.3 lets GenBio add terms
separately.

### How that maps onto distillation

| use | as the text reads | in `mpd` |
|---|---|---|
| AIDO outputs to train ESM2, SaProt, ProSST or any other non-GenBio model | falls under 1.2.2 (a) and (b); the exception names only the GenBio Materials and their Derivative Works | blocked |
| publishing AIDO outputs, caches or target tables made from them | Outputs: the same restriction, non-commercial only | nothing is shipped |
| AIDO outputs to train an adapter on the same AIDO checkpoint | inside the exception if the adapter is the GenBio Materials or a Derivative Work of them | not implemented |
| outputs of a non-GenBio model to train an AIDO model | clause 1.2.2 limits the use of AIDO's outputs, not other models' outputs | not implemented |
| commercial use, or offering AIDO outputs as a service | not allowed | not offered |

Two questions this summary cannot settle, because only GenBio can: whether an adapter file that stays separable from the base weights is a
"Derivative Work" (clause 8.4 excludes works that "remain separable from, or merely link (or bind by name) to the interfaces of" the GenBio
Materials), and what "similar" covers. If your use depends on either, ask GenBio and keep the answer in writing.

### If you use AIDO yourself, outside `mpd`

By using or distributing any part of the GenBio Materials you agree to the licence. In short, you would have to:

- use the materials, and everything you make with them, only for non-commercial purposes;
- not use their outputs to train, improve or distil other similar models, apart from the GenBio Materials themselves and their Derivative Works;
- if you distribute the GenBio Materials or a Derivative Work (for example an adapter): include a copy of the licence, keep the "Notice" text
  file with the attribution sentence "This is licensed under the GenBio AI Community License Agreement, Copyright © GENBIO.AI, INC. All Rights
  Reserved", state what you changed, and display "Powered by GenBio AI" on the related website, user interface, blog post, about page or
  documentation (clause 2);
- comply with export-control rules (clause 4), accept that the materials are provided as is (clause 5), and accept California law and
  courts (clause 7.3).

The licence is non-transferable and non-sublicensable (clause 1.1), so this repository does not redistribute any GenBio file; you obtain
them from GenBio.

### Adding AIDO later

A later release might add AIDO-to-AIDO adapters only. That would be a new decision by the maintainers, would not change the block above,
and would bring the Notice and "Powered by GenBio AI" obligations to that release. A model that you add yourself with a similar clause
should get the same lock: [ADDING_A_MODEL.md](ADDING_A_MODEL.md#output-family-locks).

## How this page was checked

On 2026-11-09, anonymously and without a token: the ProteinGym licence from the Zenodo record's metadata (version 1.3, published
2025-04-27); the VenusMutHub licence tag and head revision from the Hugging Face API (the head was still `08a27d07`); the seven model
licences from the Hugging Face API (all seven repositories public, ungated and `mit`, while the pins were rebuilt with
`scripts/maintainers/pin_models.py`); the GenBio licence file by hashing it; the package licences from installed package metadata. The Foldseek and SaProt
repository licences were read from their GitHub pages during the SaProt work ([SAPROT.md](SAPROT.md)). Nobody has asked a lawyer.
