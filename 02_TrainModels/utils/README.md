# Shared utilities

The utility package provides operational behavior shared across all phases.

- `args.py`: reusable argument groups and readable hyphenated CLI aliases.
- `device.py`: seeding, worker seeding, deterministic kernels, and
  CPU/CUDA/MPS selection.
- `io.py`: UTF-8 JSON loading/saving and directory creation.
- `logging.py`: console, schema-stable CSV, and optional TensorBoard scalars.
- `sched.py`: none, cosine, and plateau learning-rate schedules.
- `visuals.py`: maximum projections, binary/continuous comparison panels, and
  multi-angle 3D context renders.

`RunLogger` extends an existing CSV header if new scalar keys appear and closes
TensorBoard resources deterministically. Visualization functions accept
channel-first tensors, detach them to CPU, close figures after saving, and
bound dense 3D displays to control memory.

These helpers do not contain experiment-specific paths. Callers supply all
input, output, and configuration locations from the repository root.
