# Bankruptcy-Prediction

Company bankruptcy prediction from financial indicators using eight scikit-learn classifiers, with exploratory analysis, principal component analysis (PCA), and hyperparameter tuning.

This is a notebook-based experiment. The repository includes the dataset and saved notebook outputs, but no exported model, inference service, dependency lockfile, or automated tests. The reported results below are taken from the committed notebook and have not been reproduced independently.

## Repository contents

| File | Purpose |
| --- | --- |
| [Bankruptcy_identification.ipynb](Bankruptcy_identification.ipynb) | Data exploration, preprocessing, classifier comparisons, PCA, tuning, and evaluation plots |
| [data.csv](data.csv) | Financial indicators and the binary `Bankrupt?` target |
| [README.md](README.md) | Workflow, setup, results, and known limitations |

## Dataset

The bundled CSV contains **6,819 rows and 96 columns**: the target plus 95 financial indicators covering profitability, debt, cash flow, liquidity, growth, and turnover.

| Target value | Interpretation | Rows | Share |
| --- | --- | ---: | ---: |
| `0` | Non-bankrupt | 6,599 | 96.77% |
| `1` | Bankrupt | 220 | 3.23% |

The CSV has no empty fields. Many feature names contain a leading space; the first workflow explicitly drops `' Net Income Flag'`, which is constant at `1`. Preserve the exact column names when using the existing code. Although a notebook comment describes the data as min-max scaled, some feature values exceed 1; both workflows apply `StandardScaler`.

The repository does not document the original dataset source, collection period, or reuse terms.

## Getting started

Use a Python 3 environment with Jupyter and the libraries imported by the notebook. No package versions are pinned, so the following is a suggested setup rather than a verified reproduction environment:

```bash
git clone https://github.com/kanchanpotdar/Bankruptcy-Prediction.git
cd Bankruptcy-Prediction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install jupyter numpy pandas matplotlib seaborn plotly scikit-learn
jupyter notebook Bankruptcy_identification.ipynb
```

Before running locally, change the first data-loading cell from:

```python
df = pd.read_csv("/content/data.csv")
```

to:

```python
df = pd.read_csv("data.csv")
```

The original path targets a Google Colab session; in Colab, upload `data.csv` to `/content`. The final tuning cell separately reads the relative path `data.csv`, so its working directory must also contain the file.

Run the notebook cells in order to explore both workflows. The final tuning cell can also run independently: it imports its own dependencies and reloads the data. It performs five-fold grid searches for eight classifiers with `n_jobs=-1`, which can consume substantial CPU time. The final cell overwrites variables from the earlier experiment; rerun the earlier preprocessing cells before revisiting its model evaluation.

## Implemented workflows

### 1. Original features versus 10-component PCA

1. Load the CSV and inspect class counts, correlations, descriptive statistics, data types, and unique-value counts.
2. Drop the constant `' Net Income Flag'` column, leaving 94 predictors.
3. Split 70% training / 30% test with `random_state=1` and shuffling, without stratification: 4,773 training rows and 2,046 test rows.
4. Fit `StandardScaler` on training data and transform both partitions.
5. Train eight default classifiers on the standardized predictors and record test accuracy.
6. Fit a 10-component PCA on training data, transform both partitions, and retrain the same classifier families.
7. Compare accuracies and evaluate the PCA models using precision, recall, F1, and ROC-AUC where `predict_proba` is available; plot ROC curves.

The saved explained-variance ratios sum to approximately **51.01%** for the 10 components. This workflow does not retain 95% of the variance.

### 2. PCA retaining 95% variance and grid search

The final substantive code cell starts a separate experiment:

1. Reload all 95 predictors, including the constant flag, and fill missing values using column medians.
2. Standardize the full feature matrix and apply `PCA(n_components=0.95)`.
3. Make a stratified 70% / 30% split with `random_state=42`.
4. Evaluate default models, then tune each using `GridSearchCV(cv=5, scoring='roc_auc', n_jobs=-1)`.
5. Compare default and tuned test metrics and plot ROC-AUC before and after tuning.

The linear SVM uses `CalibratedClassifierCV(estimator=LinearSVC())`; the RBF SVM enables `probability=True`. The grids vary regularization, neighbor settings, tree depth and size, neural-network architecture, and boosting parameters. The notebook does not print the selected hyperparameters or final PCA component count.

**Evaluation limitation:** this second workflow fits the imputer statistics, scaler, and PCA before splitting. Test data therefore influence preprocessing, and cross-validation also receives already transformed features. Its scores are exploratory and need confirmation with preprocessing fitted inside a training-only pipeline and each cross-validation fold.

## Saved results

### Original features and 10-component PCA

Test accuracies printed in the first experiment:

| Classifier | Original features | 10-component PCA |
| --- | ---: | ---: |
| Logistic Regression | 96.48% | 96.24% |
| K-Nearest Neighbors | 96.48% | 96.29% |
| Decision Tree | 95.50% | 94.72% |
| Random Forest | 96.63% | 96.48% |
| Linear SVM | 96.29% | 96.48% |
| RBF SVM | 96.58% | 96.63% |
| Neural Network (MLP) | 95.94% | 96.29% |
| Gradient Boosting | 96.04% | 96.43% |

The saved PCA metrics imply 71 positive cases among 2,046 test rows. Always predicting non-bankrupt would achieve **96.53%** accuracy on that partition (a baseline calculated from that implied support). The RBF SVM's 96.63% accuracy accompanies just **2.82% bankruptcy recall**, illustrating why accuracy alone is insufficient.

The first workflow does not calculate ROC-AUC for either SVM because those instances lack `predict_proba`. Its printed observation that RBF SVM has the highest ROC-AUC is unsupported by its evaluation output.

### Tuned models in the separate 95%-variance PCA experiment

These are the saved post-tuning metrics from the final grid-search cell. Precision, recall, and F1 refer to the bankrupt class (`1`); accuracy is shown as a percentage and other metrics on a 0–1 scale.

| Classifier | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 96.68% | 0.4375 | 0.1061 | 0.1707 | 0.9149 |
| K-Nearest Neighbors | 96.87% | 0.6000 | 0.0909 | 0.1579 | 0.7657 |
| Decision Tree | 96.63% | 0.4211 | 0.1212 | 0.1882 | 0.8703 |
| Random Forest | 96.97% | 1.0000 | 0.0606 | 0.1143 | 0.8782 |
| Calibrated Linear SVM | 96.43% | 0.3158 | 0.0909 | 0.1412 | 0.9255 |
| RBF SVM | 96.43% | 0.2667 | 0.0606 | 0.0988 | 0.8668 |
| Neural Network (MLP) | 96.33% | 0.3784 | 0.2121 | 0.2718 | 0.8356 |
| Gradient Boosting | 96.53% | 0.4000 | 0.1515 | 0.2198 | 0.9161 |

Among these saved tuned results, **calibrated Linear SVM has the highest ROC-AUC (0.9255)**, while **MLP has the highest recall and F1**. The notebook's closing markdown names Gradient Boosting as the ROC-AUC winner, which conflicts with its own results table. Tuning does not improve every metric for every model.

The stratified test partition contains 66 bankrupt cases out of 2,046 rows; an always-non-bankrupt baseline is **96.77%** accurate. Tuned recall ranges from approximately 6% to 21%, so most bankrupt cases remain undetected at the models' default decision thresholds. The two experiments use different splits, features, and PCA settings and should be interpreted separately.

## Reproducibility and remaining work

- Saved outputs are historical results, not a fresh benchmark. Dependency versions are unspecified, and several stochastic estimators have no fixed `random_state`.
- The second workflow needs training-only preprocessing within a scikit-learn pipeline before its evaluation can support reliable generalization claims.
- No class weighting, resampling, or decision-threshold optimization is implemented. The strong class imbalance makes bankruptcy recall and precision essential alongside ROC-AUC.
- There is no exported model, persisted preprocessing pipeline, prediction API, or deployment code. Plots and result tables are displayed in the notebook.
- SHAP/LIME explainability is suggested in notebook text but not implemented. No license file is included in the repository.
