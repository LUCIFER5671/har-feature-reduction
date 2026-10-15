# HAR Feature Reduction & Unsupervised Learning

Unsupervised learning and feature-reduction lab on the UCI *Human Activity Recognition Using Smartphones*
dataset (561 features, 30 subjects, 6 activities). Everything lives in one notebook,
`har_feature_reduction.ipynb`, and a PDF export of the executed notebook is included as
`har_feature_reduction.pdf`.

| Phase | Content |
|---|---|
| 1 - Setup | Locate and load the data, subject-wise train/dev split, scaling, cache |
| 2 - EDA | Activity counts, PCA explained variance and elbow, PC1/PC2 scatter, PC1-PC3 loadings |
| 3 - Clustering | K-Means (silhouette and inertia vs k), DBSCAN on 3 PCs and on all 561 features |
| 4 - Feature reduction | PCA transform vs random-forest feature selection |
| 5 - Classification | Logistic regression vs number of features, efficiency, final test |
| Report | Written analysis under Phases 2, 3 and 5, reflection, AI-use statement |

## 1. Get the code and set up Python

Requires Python 3.10+ (developed with 3.11).

```bash
git clone https://github.com/LUCIFER5671/har-feature-reduction.git
cd har-feature-reduction
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Download the dataset

The dataset is not in the repository (it is gitignored).

1. Download it from the UCI Machine Learning Repository:
   <https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones>
   (button **Download**, file `human+activity+recognition+using+smartphones.zip`, about 60 MB).
2. Extract that zip **inside the project folder**. It contains a second archive, `UCI HAR Dataset.zip`;
   extract that one too.

The notebook searches the project folder for the directory that contains `train/`, `test/`, `features.txt`
and `activity_labels.txt`, so the exact nesting does not matter. With the default extraction it ends up at:

```
human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/
```

## 3. Run the notebook

Open `har_feature_reduction.ipynb` in Jupyter or VS Code, select the `.venv` kernel and run all cells, or run it
headless (takes a few minutes):

```bash
jupyter nbconvert --to notebook --execute --inplace har_feature_reduction.ipynb
```

Run the cells in order: later phases use objects created by earlier ones (for example the Phase 2 PCA fit and
the Phase 4 feature ranking).

### Export to PDF (optional)

`jupyter nbconvert --to pdf` needs a LaTeX installation. Without LaTeX, export to HTML and print that to PDF with
a Chromium-based browser, e.g. Microsoft Edge on Windows:

```bash
jupyter nbconvert --to html har_feature_reduction.ipynb
msedge --headless --disable-gpu --no-pdf-header-footer --print-to-pdf=har_feature_reduction.pdf har_feature_reduction.html
```

## Outputs

- `figures/` - every figure as PNG (dpi=200)
- `results/` - every numeric table and score as CSV; all wall-clock timings are collected in `results/timings.csv`
  (the only result file that changes between re-runs)
- `cache/har_processed.npz` - standardized train/dev/test arrays written by Phase 1 (regenerated on each run,
  gitignored)

## Protocol

- `random_state=42` everywhere.
- The provided training set is split **by subject** (GroupShuffleSplit, ~20% of subjects) into train / dev;
  all exploration and model selection use only these.
- `StandardScaler` is fitted on train only and applied to train, dev and test.
- The provided test set is loaded, scaled with the train-fitted scaler and cached in Phase 1 (its class counts
  appear in the split table). No model is trained or evaluated with it until the single final evaluation in
  Phase 5 (section 5.6).
