# External research artifacts

Large research artifacts are released separately from this source repository.
This separation keeps Git history reviewable and prevents software licensing
from being confused with source-data, derived-data, or model-weight terms.

## Release records

The external research artifacts are distributed through the public Cloudflare
R2 bucket documented below. The authoritative release inventory is the
`artifacts_manifest.json` file published in that bucket. It records the exact
uploaded prefixes, expected local restore paths, file counts, byte totals, key
checkpoint URLs, and S3-compatible restore commands.

| Artifact | Repository location after restore | Release record |
|---|---|---|
| Generated thesis dataset | `00_Data/02_GeneratedDatasets/dataset_thesis/` | R2 artifact manifest |
| Dataset reports and fixed split | Adjacent to `dataset_thesis/` | R2 artifact manifest |
| Final model checkpoint bundle | `02_TrainModels/outputs/thesis_runs/` | R2 artifact manifest |
| Compact experiment reports | `02_TrainModels/outputs/thesis_runs/` | R2 artifact manifest |

## Public artifacts

Data and trained model outputs are hosted on Cloudflare R2.

Public development URL:

<https://pub-974ec55907ff42e581f011cd95ef519e.r2.dev>

Dataset root:

<https://pub-974ec55907ff42e581f011cd95ef519e.r2.dev/00_Data/>

Model outputs root:

<https://pub-974ec55907ff42e581f011cd95ef519e.r2.dev/02_TrainModels/outputs/>

R2 public buckets do not expose directory listings. Use the artifact manifest
for exact prefix names, expected restore locations, file counts, byte totals,
key checkpoint URLs, and restore commands.

## Download helper

The repository root contains `download_artifacts.py`, which reads
`artifacts_manifest.json` from that bucket by default. The manifest records
uploaded R2 prefixes, local restore paths, file counts, byte totals, selected
key checkpoint URLs, public file-index locations, and S3-compatible restore
commands.

Verify that the public manifest is reachable and list available artifacts:

```bash
python download_artifacts.py --list
```

Restore all published artifacts through public HTTP object URLs:

```bash
python download_artifacts.py --public-download
```

Restore only one artifact prefix:

```bash
python download_artifacts.py --only generated_thesis_dataset --public-download
```

Public R2 buckets do not expose directory listings. The public download mode
therefore uses newline-delimited JSON file indexes referenced by
`artifacts_manifest.json`; each index lists exact object keys, target paths,
byte sizes, and optional SHA-256 hashes. If an index object is missing from R2,
the selected prefix cannot be restored over anonymous HTTP.

The S3-compatible restore path remains available for maintainers with
Cloudflare R2 credentials. Do not commit those credentials. Use the exact
repository casing, for example `00_Data/` rather than `00_data/`.

Maintainers can rebuild the public HTTP indexes directly from the R2 object
listing:

```bash
export R2_ENDPOINT="https://adecb175189b1b6a8ad72f5a58650667.r2.cloudflarestorage.com"
export R2_BUCKET="genaiforearlystageurbanmassing"
python 00_Data/build_public_file_indexes.py
aws s3 cp artifacts_manifest.json "s3://$R2_BUCKET/artifacts_manifest.json" \
  --endpoint-url "$R2_ENDPOINT" --region auto --content-type "application/json"
aws s3 cp .artifacts/file_indexes/ "s3://$R2_BUCKET/file_indexes/" \
  --recursive --endpoint-url "$R2_ENDPOINT" --region auto \
  --content-type "application/x-ndjson"
```

The manifest currently declares these restore commands:

```bash
export R2_ENDPOINT="https://adecb175189b1b6a8ad72f5a58650667.r2.cloudflarestorage.com"
export R2_BUCKET="genaiforearlystageurbanmassing"
aws s3 sync "s3://$R2_BUCKET/00_Data" "00_Data" --endpoint-url "$R2_ENDPOINT" --region auto
aws s3 sync "s3://$R2_BUCKET/02_TrainModels/outputs" "02_TrainModels/outputs" --endpoint-url "$R2_ENDPOINT" --region auto
```

## Dataset artifact contents

The restored generated dataset should contain:

```text
dataset_thesis/
├── <parcel_id>/
│   ├── <parcel_id>.npz
│   └── <parcel_id>.json
├── report.json
├── report_split.json
├── validation.json
├── creation_config.json
├── download_manifest.json
└── SHA256SUMS
```

Keep raw Berlin inputs outside the generated dataset directory. They can be
retrieved from the official providers with
`01_CreateDataset/dataset_cli.py download`. The source manifest records the
exact retrieval URLs, dates, sizes, and checksums needed to identify the input
snapshot.

NPZ files are already compressed. Preserve the directory structure shown above
when moving the dataset between local storage, R2, or any later research-data
archive.

## Model artifact contents

The restored model outputs should include four selected checkpoints:

1. the final unconditional U-Net;
2. the final conditional U-Net;
3. the VAE required by latent diffusion; and
4. the final latent diffusion model.

Each checkpoint directory must also contain its resolved `config.json`, final
metrics, evaluation summaries, conditioning statistics where applicable, and
the checkpoint-selection rule. Preserve the paths used by the versioned thesis
configs so inference commands work without manual path edits.

## Verification after download

After restoring the artifacts, verify the dataset and inspect the available
model commands. The validation command fails with `dataset root not found` when
`00_Data/02_GeneratedDatasets/dataset_thesis/` has not yet been restored from
R2.

```bash
python 01_CreateDataset/dataset_cli.py validate \
  --root 00_Data/02_GeneratedDatasets/dataset_thesis \
  --check-nan

python 02_TrainModels/train_cli.py status
python 03_visuals/visuals_cli.py --help
```

Licensing boundaries for source code, thesis text, figures, inputs, generated
data, checkpoints, and public downloads are described in
[DATA_LICENSE.md](DATA_LICENSE.md). The R2 bucket is a distribution endpoint,
not a commercial-use or production-deployment license grant.
