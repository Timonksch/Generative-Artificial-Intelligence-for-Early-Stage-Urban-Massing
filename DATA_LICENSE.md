# Data and Artifact Licensing

This document separates the license for the repository source code from the
licenses and terms that apply to geospatial input data, generated voxel
datasets, trained model artifacts, reports, and public downloads.

The repository `LICENSE` applies only to source code authored for this project.
It does not automatically apply to input data, generated data, map tiles,
fonts, trained model weights, rendered figures, or other third-party artifacts.

## Source Code

Project source code is licensed under the MIT License in `LICENSE`.

The following repository contents are treated as source-code or documentation
materials covered by that license unless stated otherwise:

- Python source files and command-line tools;
- experiment configuration files;
- repository documentation;
- tests and test fixtures authored for this project; and
- small metadata examples used to validate repository behavior.

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

The generated dataset may be redistributed only together with:

- the applicable source-data license notice;
- the exact creation configuration;
- the source `download_manifest.json`;
- dataset-level SHA-256 checksums;
- the fixed split manifest used for reported experiments; and
- the validation report for the released archive.

The distribution terms for the generated dataset follow the upstream
source-data license identified above and the provenance requirements in this
file. Do not infer dataset licensing from the repository MIT License.

## Model Checkpoints and Experiment Outputs

Trained model weights, resolved training configurations, metric summaries,
evaluation outputs, and qualitative prediction artifacts are research artifacts
released separately from Git.

These artifacts are not covered by the repository MIT License. They may be
used, copied, and redistributed for academic review, research, and
reproducibility purposes under the same provenance and source-license boundary
as the generated voxel dataset.

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
dataset, not as MIT-licensed source code.

## User Responsibility

Users are responsible for:

- checking the current provider terms before downloading a new source snapshot;
- preserving source-data notices and provenance when redistributing generated
  datasets or model artifacts;
- checking artifact sizes and validating restored datasets before use; and
- ensuring that any downstream publication or reuse complies with the relevant
  source-data and artifact-release terms.

No data or artifact license should be inferred from the presence of metadata,
metadata, example paths, download scripts, or public bucket URLs in this
repository.
