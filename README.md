# HAR Feature Reduction & Unsupervised Learning

Unsupervised learning lab on the UCI *Human Activity Recognition Using Smartphones* dataset
(561 features, 30 subjects, 6 activities). Everything lives in one notebook:
`har_feature_reduction.ipynb`.

## Setup

Requires Python 3.10+ (developed with 3.11).

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

The dataset must be extracted at:

```
human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/
```

## Run

Open the notebook in Jupyter or VS Code (select the `.venv` kernel) and run all cells, or run it headless:

```bash
jupyter nbconvert --to notebook --execute --inplace har_feature_reduction.ipynb
```

## Outputs

- `figures/` — every figure as PNG (dpi=200)
- `results/` — every numeric table, score and timing as CSV
- `cache/har_processed.npz` — standardized train/dev/test arrays written by Phase 1 (regenerated on each run)

## Protocol

- `random_state=42` everywhere.
- The dataset folder is located automatically inside the project; nothing is downloaded or extracted.
- The provided training set is split **by subject** (GroupShuffleSplit, ~20% of subjects) into train / dev;
  all exploration and model selection use only these.
- `StandardScaler` is fitted on train only and applied to train, dev and test.
- The provided test set is scaled and cached in Phase 1 but used exactly once, in the final evaluation (Phase 5).
