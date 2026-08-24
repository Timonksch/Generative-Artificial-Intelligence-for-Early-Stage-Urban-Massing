# Licensing, Data, and Artifact Terms

This document separates the terms for repository source code, thesis text,
documentation, figures, geospatial input data, generated voxel datasets,
trained model artifacts, reports, and public downloads.

The repository `LICENSE` applies only to source code authored for this project.
It does not apply to the thesis manuscript, written documentation, rendered
figures, diagrams, input data, generated data, map tiles, fonts, trained model
weights, or other third-party artifacts.

This licensing statement defines the applicable terms for the public thesis
submission state dated September 1, 2026, and later distributions made from it.

## Source Code

Project source code is licensed under the GNU Affero General Public License
version 3.0 only (`AGPL-3.0-only`) in `LICENSE`.

The following repository contents are treated as source-code materials covered
by that license unless stated otherwise:

- Python source files and command-line tools;
- experiment configuration files;
- tests and test fixtures authored for this project;
- source-code comments and inline developer documentation; and
- small metadata examples used to validate repository behavior.

The AGPL source-code license permits study, copying, modification, and
redistribution under its reciprocal terms. Modified versions distributed or
made available as network services must comply with the source-availability
requirements of the AGPL.

## Thesis, Documentation, and Figures

The thesis manuscript, written repository documentation, rendered figures,
diagrams, explanatory images, and presentation-style material authored for this
project are licensed under:

**Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
(`CC BY-NC-ND 4.0`)**

<https://creativecommons.org/licenses/by-nc-nd/4.0/>

This permits academic reading, sharing, and citation with attribution. It does
not grant permission for commercial use or distribution of modified versions.
For permissions beyond these terms, contact the author.

The public thesis PDF included in the repository is covered by this section,
not by the AGPL source-code license.

## Official Berlin Input Data

The dataset pipeline uses public geospatial data from the State of Berlin:

| Input | Local role | Documented source license |
|---|---|---|
| Berlin LoD1 building model | building footprints and heights | Data licence Germany - Zero - Version 2.0 |
| ALKIS Berlin parcels | cadastral parcel geometries | Data licence Germany - Zero - Version 2.0 |
| Detailnetz Berlin | street-network context | Data licence Germany - Zero - Version 2.0 |

The provider metadata checked for this project identifies these sources as
available under:

**Data licence Germany - Zero - Version 2.0**

<https://www.govdata.de/dl-de/zero-2-0>

Source endpoints, retrieval commands, verified counts, and snapshot notes are
documented in `00_Data/01_InputData/README.md`. Provider metadata and service
contents may change over time. Anyone creating a new input snapshot should
verify the current provider terms and keep the generated
`download_manifest.json` with the archived dataset.

## Generated Voxel Dataset

The generated voxel dataset is a derived research artifact produced from the
official Berlin input data and project-authored preprocessing code. It includes
voxel arrays, sample metadata, dataset reports, split manifests, validation
reports, and provenance files.

To the extent rights in the generated dataset are held by the project author,
the generated dataset is made available only for academic review, research, and
reproducibility of the associated master's thesis. No permission is granted for
commercial use, production deployment, resale, or redistribution without the
provenance materials listed below and without respecting all upstream data
terms. This restriction does not limit any rights that users may hold
independently under the official upstream Berlin data licenses.

The generated dataset may be redistributed only together with:

- the applicable source-data license notice;
- the exact creation configuration;
- the source `download_manifest.json`;
- dataset-level SHA-256 checksums;
- the fixed split manifest used for reported experiments; and
- the validation report for the released archive.

The distribution terms for the generated dataset follow the upstream
source-data license identified above and the provenance requirements in this
file. Do not infer dataset licensing from the repository AGPL source-code
license.

## Model Checkpoints and Experiment Outputs

Trained model weights, resolved training configurations, metric summaries,
evaluation outputs, and qualitative prediction artifacts are research artifacts
released separately from Git.

These artifacts are not covered by the repository AGPL source-code license.
To the extent rights in these artifacts are held by the project author, they
may be used, copied, and redistributed only for academic review, research, and
reproducibility purposes under the same provenance and source-license boundary
as the generated voxel dataset. No permission is granted for commercial use,
production deployment, resale, or use as part of a commercial service without
prior written permission from the author.

Any redistribution of model checkpoints or experiment outputs must include:

- this `DATA_LICENSE.md` file;
- the artifact manifest containing restore paths, file counts, byte totals,
  and key artifact URLs;
- the resolved training configuration for each released run;
- the dataset provenance needed to identify the training data; and
- the source-data license notice for the official Berlin input data.

The model artifacts are provided without warranty. No permission is granted to
use them in a way that removes or obscures the dataset provenance, source-data
license notice, or artifact manifest.

## Public Cloudflare R2 Artifacts

Public development artifacts are hosted on Cloudflare R2:

<https://pub-974ec55907ff42e581f011cd95ef519e.r2.dev>

The R2 bucket is a distribution location, not a license grant. The authoritative
artifact inventory is `artifacts_manifest.json`, which records restore paths,
file counts, byte totals, key artifact URLs, and restore commands. See
`ARTIFACTS.md` and `download_artifacts.py` for the download and verification
workflow.

## Examples and Smoke Data

Small committed example samples are included only for tests, demonstrations,
and interface validation. They are derived from the same source-data pipeline
and should be treated under the same data-license boundary as the generated
dataset, not as AGPL-licensed source code.

## User Responsibility

Users are responsible for:

- checking the current provider terms before downloading a new source snapshot;
- preserving source-data notices and provenance when redistributing generated
  datasets or model artifacts;
- checking artifact sizes and validating restored datasets before use; and
- ensuring that any downstream publication or reuse complies with the relevant
  source-data and artifact-release terms.

No data or artifact license should be inferred from the presence of metadata,
example paths, download scripts, or public bucket URLs in this repository.
