# Thesis experiment sequence

The 23 configurations encode the experiments reported for the three thesis
phases. Their numeric prefixes define the intended experimental order. Files
ending in `final` are the selected long reruns; selection was based on the
corresponding validation protocol, not test performance.

All configurations share the same generated dataset and split manifest. Keep
those two artifacts fixed when reproducing comparisons. Phase-III diffusion
runs additionally share the selected frozen VAE checkpoint.

Detailed phase documentation:

- [`phase1/README.md`](phase1/README.md): unconditional U-Net experiments;
- [`phase2_cond/README.md`](phase2_cond/README.md): controlled U-Net experiments;
- [`phase3_ldm/README.md`](phase3_ldm/README.md): VAE and diffusion sequence.

List the exact versioned files with:

```bash
python 02_TrainModels/train_cli.py configs
```
