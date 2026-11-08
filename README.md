# Breast Tumour Classification: Benign vs Malignant

## a. Problem statement

Fine-needle aspiration (FNA) of a breast mass produces a cell sample that a pathologist examines under a microscope. The goal of this project is to predict, from 30 numeric measurements of the cell nuclei in a digitised FNA image, whether the mass is **malignant (M)** or **benign (B)**.

This is a binary classification problem. Five classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes and Support Vector Machine) are trained on the same data, evaluated on the same held-out test split with six metrics, and served through a Streamlit app that loads the saved models and scores an uploaded test CSV without retraining.

Malignant is treated as the positive class, so Recall answers the clinically important question: of the cancers in the test set, how many did the model catch?

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | UCI Machine Learning Repository (also mirrored on Kaggle as *uciml/breast-cancer-wisconsin-data*) |
| Instances | 569 |
| Features | 30, all real-valued |
| Target | `diagnosis`: M = malignant (212, 37.3%), B = benign (357, 62.7%) |
| Missing values | None |

Ten characteristics are computed for every nucleus in the image: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. For each characteristic, three summary values are recorded over all nuclei in the image: the mean (`_mean`), the standard error (`_se`) and the mean of the three largest values (`_worst`). That gives 10 × 3 = 30 features. The `id` column is dropped before training.

The features are on very different scales (area is in the hundreds to thousands while smoothness is around 0.1) and several are strongly correlated; for example radius, perimeter and area correlate above 0.98, and 21 feature pairs have |r| > 0.9. Both facts influence the results below.

**Split used:** stratified 80/20 train–test split (`random_state=47`): 455 training rows and 114 test rows (72 B, 42 M). The 114 test rows are saved as `test_data.csv`.

## c. GitHub repository link

**Repository:** https://github.com/Anishpodder6/breast-cancer-classifier.git

**Live Streamlit app:** https://breast-cancer-classifier-avn3xjrghtzrajvnrdbmzf.streamlit.app/

```
breast-cancer-classifier/
├── app.py                 Streamlit app (loads saved models, scores uploaded CSV)
├── utils.py               Upload validation and metric computation used by the app
├── train_models.py        Training, hyper-parameter search, evaluation, saving
├── requirements.txt
├── README.md
├── test_data.csv          Held-out test split with the diagnosis column
├── data/
│   └── wdbc.csv           Full dataset used for training
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── support_vector_machine.joblib
│   ├── metadata.json      Feature order, label mapping, tuned hyper-parameters
│   └── metrics.csv        Test-set metrics table
└── .streamlit/config.toml Theme
```

**Reproduce:**

```bash
pip install -r requirements.txt
python train_models.py      # trains, prints metrics, writes model/ and test_data.csv
streamlit run app.py
```

### Approach

- Each model is tuned with `GridSearchCV` using stratified 5-fold cross-validation on the training data only, optimising F1 for the malignant class. The test split is never seen during tuning.
- Logistic Regression, KNN and SVM are wrapped in a `Pipeline` with `StandardScaler`, so the scaler is fitted on training data and saved together with the model. Decision Tree and Gaussian Naive Bayes do not depend on feature scale and are used on raw features.
- AUC is computed from predicted probabilities. For the SVM, the decision-function value is used, which gives the same ROC ranking without the extra cost of probability calibration.

| Model | Tuned hyper-parameters |
|---|---|
| Logistic Regression | C = 5 |
| Decision Tree | criterion = entropy, max_depth = 3, min_samples_leaf = 1 |
| KNN | k = 11, weights = uniform |
| Naive Bayes (Gaussian) | var_smoothing = 1e-9 |
| SVM (RBF kernel) | C = 5, gamma = 0.01 |

## d. Models used

### Comparison table (held-out test set, 114 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | **0.9737** | 0.9808 | **0.9756** | **0.9524** | **0.9639** | **0.9433** |
| Decision Tree | 0.9386 | 0.9516 | 0.9070 | 0.9286 | 0.9176 | 0.8689 |
| KNN | 0.9561 | 0.9818 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9298 | 0.9835 | 0.9250 | 0.8810 | 0.9024 | 0.8483 |
| Support Vector Machine | 0.9649 | **0.9868** | 0.9524 | **0.9524** | 0.9524 | 0.9246 |

Confusion matrices on the test set (TN / FP / FN / TP, where FN = malignant predicted as benign):

| Model | TN | FP | FN | TP | Total errors |
|---|---|---|---|---|---|
| Logistic Regression | 71 | 1 | 2 | 40 | 3 |
| Decision Tree | 68 | 4 | 3 | 39 | 7 |
| KNN | 71 | 1 | 4 | 38 | 5 |
| Naive Bayes | 69 | 3 | 5 | 37 | 8 |
| Support Vector Machine | 70 | 2 | 2 | 40 | 4 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best on Accuracy (0.9737), Precision, F1 and MCC (0.9433), with only 3 errors out of 114. Once the features are standardised, the two classes are close to linearly separable, so a linear boundary with mild regularisation (C = 5) is enough. The large-scale features (area, perimeter) would otherwise dominate, so scaling is essential here. |
| Decision Tree | Weakest overall: lowest AUC (0.9516) and second-lowest MCC (0.8689), with 7 errors spread across both classes. The tuned tree has depth 3, and `area_worst` alone accounts for about 68% of its importance, so it splits on a few thresholds and produces axis-parallel, step-shaped boundaries. Its AUC is low because a depth-3 tree has only a handful of leaves and therefore only a few distinct probability values to rank samples with. Its strength is interpretability: the rules can be read directly. |
| KNN | High precision (0.9744, only 1 false alarm) but lower recall (0.9048): it missed 4 of the 42 malignant cases. With k = 11 and uniform voting, borderline malignant samples sitting near the larger benign group get outvoted. Distance in 30 dimensions is also inflated by the many correlated size features, which effectively count the same information several times. |
| Naive Bayes | Lowest Accuracy (0.9298), Recall (0.8810) and MCC (0.8483), missing 5 cancers. The independence assumption is clearly violated, since radius, perimeter and area are almost perfectly correlated, so the model double-counts the same evidence and becomes over-confident. Interestingly its AUC (0.9835) is the second-highest: the ranking of samples is good, but the default 0.5 threshold on its distorted probabilities places the boundary in the wrong spot. |
| Support Vector Machine | Highest AUC (0.9868) and tied-best Recall (0.9524) with Logistic Regression, with 4 errors split evenly (2 FP, 2 FN). The selected gamma of 0.01 is small, which makes the RBF boundary smooth and close to linear; this agrees with Logistic Regression doing so well and suggests little non-linearity is needed for this dataset. |
| **Overall winner for this dataset** | **Logistic Regression.** It leads on Accuracy, F1 and MCC (the most balanced single metric for an imbalanced binary problem) and ties for the best Recall, while also being the simplest and most interpretable model. The SVM is a very close second and has the best AUC. With 114 test rows, one sample is worth about 0.9% accuracy, so the gap between the two is a single prediction and they should be considered practically equal; Logistic Regression is preferred for its simplicity and its directly interpretable coefficients. |

## Streamlit app features

- **CSV upload** of test data (the bundled `test_data.csv` can be downloaded from the sidebar and uploaded back, or used directly with a toggle).
- **Model selection dropdown** for the five saved classifiers.
- **Evaluation metrics** (Accuracy, AUC, Precision, Recall, F1, MCC) for the selected model.
- **Confusion matrix and classification report**, plus a ROC curve.
- **Comparison tab** scoring all five models on the uploaded file, with a metric chart and a CSV download.
- **Row-level predictions**, with a filter to show only misclassified rows.

The uploaded file must contain the 30 feature columns and the `diagnosis` column (M/B, malignant/benign or 1/0 are accepted). The app validates the columns, loads the saved models from `model/`, and only calls `predict`; it never refits on the uploaded data.
