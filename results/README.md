# Results

Training writes loss and perplexity curves to `results/figures/` and checkpoints
to `checkpoints/`. Both are generated artifacts and are gitignored.

Reproduce the tiny CPU baselines:

```bash
python scripts/train_lm.py --model transformer --preset tiny
python scripts/train_classifier.py --featurizer bow
```

The current reproducible numbers are reported in the top-level README. Curated
figures and headline numbers from the full GPU runs (`--preset full`) are added
here once those runs are complete.
