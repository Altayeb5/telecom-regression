# Telecom Customer Revenue Prediction

## Overview

This project develops and evaluates regression models that predict a telecom customer's `Aggregate_Total_Rev` from usage, tenure, complaints, user type, and preferred-operator information. The analysis follows an end-to-end workflow: data-quality review, leakage prevention, exploratory analysis, pipeline-based preprocessing, baseline comparison, regularized regression, cross-validation, and residual diagnostics.

The selected model is a Ridge regression with `alpha=100`, selected using five-fold cross-validation on the training data. On the held-out test set, it achieved an MAE of **487.13**, RMSE of **1,318.83**, and R² of **0.1980**.

## Business Problem

For a telecom operator, an estimate of customer revenue could support customer-value assessment, revenue planning, segmentation, and prioritization of high-value customers. Large gaps between expected and actual revenue could also flag customers worth further investigation.

These are potential applications only: the dataset is not presented as data from a specific telecom operator, and this project does not claim production deployment.

## Dataset

- **Name:** South Asian Wireless Telecom Operator (SATO 2015)
- **Source:** [Kaggle — SATO 2015](https://www.kaggle.com/datasets/mahreen/sato2015)
- **File used:** `data/raw/South Asian Wireless Telecom Operator (SATO 2015).csv`
- **Shape:** 2,000 observations and 14 original columns
- **Target:** `Aggregate_Total_Rev`

The data contains customer network age, aggregate data volume, calls, complaints, revenue-related fields, user-type fields, preferred-operator fields, and a churn-status field. The final regression problem uses telecom usage and customer characteristics while excluding revenue-derived fields and the churn outcome.

## Project Objectives

- Predict customer total revenue (`Aggregate_Total_Rev`).
- Establish whether selected telecom usage and customer characteristics improve on a mean-revenue baseline.
- Compare Multiple Linear Regression, Ridge Regression, and Lasso Regression.
- Select regularization strength through cross-validation without using the test set for tuning.
- Diagnose model behavior, especially for high-revenue customers.

## Data Leakage Prevention

The analysis explicitly investigated whether revenue-component columns could make the target prediction artificially easy. `Revenue_Sum` was created from `Aggregate_SMS_Rev`, `Aggregate_Data_Rev`, `Aggregate_ONNET_REV`, and `Aggregate_OFFNET_REV`; `Revenue_Difference` was then calculated from the target and that sum.

Although the components did not simply sum to `Aggregate_Total_Rev`, they are revenue-related variables and were excluded from modeling to avoid using information too close to the outcome being predicted. The following fields were not used as model inputs:

- `Aggregate_SMS_Rev`, `Aggregate_Data_Rev`, `Aggregate_ONNET_REV`, and `Aggregate_OFFNET_REV` — revenue-component fields
- `Revenue_Sum` — derived revenue feature
- `Revenue_Difference` — derived using the target
- `Class` — churn outcome, not used for this revenue-prediction task

This left a deliberately limited, behavior- and customer-characteristic-based feature set.

## Exploratory Data Analysis

Key observations from the analysis:

- The target is strongly right-skewed (skewness **7.40**); revenue ranges from **4.91** to **24,438.83**, with a median of **606.58**.
- The distribution and boxplot show a long right tail and legitimate-looking high-revenue observations. They were investigated rather than automatically removed.
- `Aggregate_Calls` had the strongest simple numerical correlation with total revenue among the retained predictors (**0.54**). The retained numerical predictors did not show obvious severe multicollinearity in the correlation matrix.
- Customers classified as `3G` had higher average aggregate revenue in the categorical summaries.
- Missing values were concentrated in categorical columns: `aug_user_type` (245), `sep_user_type` (206), `aug_fav_a` (1), and `sep_fav_a` (1) before cleaning.
- Five `network_age` values were negative, and there were no duplicate rows.
- The string value `0` appeared 59 times in `aug_fav_a`; it was treated as an unknown/missing category rather than as an operator.

![Distribution of Aggregate Total Revenue](images/Distribution%20of%20Aggregate%20Total%20Revenue.png)

![Correlation Matrix — Regression Variables](images/Correlation%20Matrix%20%E2%80%94%20Regression%20Variables.png)

## Data Cleaning and Preprocessing

Data-quality decisions were made before modeling:

- Converted the five negative `network_age` values to missing values because a negative network age is not meaningful.
- Replaced the string `0` in `aug_fav_a` with a missing value.
- Preserved rows rather than deleting them; the dataset had no duplicate records.

Preprocessing was then placed inside scikit-learn pipelines, fitted only on training data:

- Numerical features: median imputation.
- Categorical features: constant imputation with `Unknown`, then one-hot encoding with unknown-category handling.
- Ridge and Lasso: numerical features were standardized after imputation.

The data was split once into 80% training (1,600 rows) and 20% test (400 rows), using `random_state=42`. The test set remained separate during preprocessing fitting and hyperparameter selection.

## Feature Selection

**Target**

`Aggregate_Total_Rev`

**Final modeling features**

| Numerical | Categorical |
|---|---|
| `network_age` | `aug_user_type` |
| `Aggregate_Data_Vol` | `sep_user_type` |
| `Aggregate_Calls` | `aug_fav_a` |
| `Aggregate_complaint_count` | `sep_fav_a` |

## Modeling Approach

### 1. Mean Baseline

The baseline predicts the training-set mean revenue for every test observation. It provides the minimum reference point for determining whether the modeling features add predictive value.

**Test results:** MAE **676.3778**, RMSE **1473.1940**, R² **-0.0007**.

### 2. Multiple Linear Regression

Multiple Linear Regression used the shared preprocessing pipeline with median imputation for numerical fields and imputation plus one-hot encoding for categorical fields.

**Test results:** MAE **511.3114**, RMSE **1346.8820**, R² **0.1635**.

### 3. Ridge Regression

Ridge Regression was tested to regularize coefficients in a feature space containing scaled numerical variables and one-hot encoded categorical variables.

**Ridge (`alpha=1`) test results:** MAE **491.2771**, RMSE **1305.9091**, R² **0.2137**.

### 4. Lasso Regression

Lasso Regression was also tested with scaled numerical features and the same categorical preprocessing.

**Lasso (`alpha=1`) test results:** MAE **491.0734**, RMSE **1306.6676**, R² **0.2127**.

### 5. Cross-Validation and Hyperparameter Tuning

`GridSearchCV` tuned each regularized model on the training data only, using five folds and `neg_root_mean_squared_error` as the scoring metric.

| Model | Alpha values tested | Selected alpha | Best CV RMSE |
|---|---|---:|---:|
| Ridge | 0.01, 0.1, 1, 10, 100, 1000 | 100 | 797.7424 |
| Lasso | 0.001, 0.01, 0.1, 1, 10, 100 | 10 | 798.7256 |

Ridge `alpha=100` was selected because it had the lowest average cross-validation RMSE among the Ridge candidates—not because of test-set performance. The test set was evaluated only after selection.

## Model Comparison

| Model | Alpha | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| Mean Baseline | — | 676.3778 | 1473.1940 | -0.0007 |
| Linear Regression | — | 511.3114 | 1346.8820 | 0.1635 |
| Ridge | 1 | 491.2771 | **1305.9091** | **0.2137** |
| Ridge (tuned; selected by CV) | 100 | **487.1287** | 1318.8336 | 0.1980 |
| Lasso | 1 | 491.0734 | 1306.6676 | 0.2127 |
| Lasso (tuned) | 10 | 489.5465 | 1322.1030 | 0.1940 |

Regularization improved all reported test metrics over ordinary Linear Regression. Ridge `alpha=100` is the selected model because cross-validation chose it without consulting the test set; it also had the lowest test MAE in this comparison. Ridge `alpha=1` happened to show the best observed test RMSE and R², but selecting it on that basis would improperly use the test set for model selection.

The selected Ridge model reduced MAE by **27.98%** relative to the mean baseline.

## Coefficient Interpretation

The selected Ridge model's largest reported coefficients included:

- `Aggregate_Calls`: **+561.59**
- `Aggregate_Data_Vol`: **+129.85**
- `network_age`: **+31.63**
- `sep_fav_a_ptcl`: **-178.47**
- `sep_fav_a_warid`: **+170.03**
- `aug_user_type_3G`: **+133.02**
- `sep_user_type_3G`: **+64.19**

The numerical inputs were standardized for Ridge, so their coefficients describe associations for approximately one-standard-deviation changes, holding other modeled variables constant. Categorical coefficients describe the fitted association for each encoded category, conditional on the other modeled variables. The notebook discusses these as comparisons with a reference category; however, its `OneHotEncoder` configuration does not drop a category, so no categorical level is formally omitted. Because these representations differ, standardized numerical and one-hot categorical coefficient magnitudes should not be treated as directly equivalent importance scores. These are model associations, not causal effects.

## Residual Analysis

For the selected Ridge model, residuals (`actual − predicted`) had a mean of **15.73**, a minimum of **-2402.91**, and a maximum of **22492.85**. Most residuals were near zero, but their spread increased at higher predicted revenue, indicating heteroscedasticity. The large positive residual shows substantial underprediction for at least one extreme high-revenue customer.

This helps explain why RMSE remains much larger than MAE: the model is more useful for typical customers than for the most extreme revenue observations.

![Ridge Regression Residuals vs Predicted Revenue](images/Ridge%20Regression-%20Residuals%20vs%20Predicted%20Revenue.png)

## Log-Target Experiment

Because the target was strongly right-skewed, the project tested a Ridge model trained on `log1p(Aggregate_Total_Rev)` with `alpha=100`. Predictions were transformed back to the original revenue scale with `expm1` before evaluation.

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Ridge, original target (`alpha=100`) | **487.1287** | **1318.8336** | **0.1980** |
| Ridge, `log1p` target (`alpha=100`) | 557.6023 | 1553.6930 | -0.1131 |

The log-target experiment performed worse on the final test set and was rejected. Compressing high revenue values during training did not improve prediction of revenue on its original scale.

## Key Findings

- The retained behavior and customer-characteristic features clearly improved on the mean baseline.
- Regularized linear models outperformed ordinary Linear Regression on this test split.
- Cross-validation selected Ridge `alpha=100`; its test MAE was the lowest among the listed models.
- Ridge `alpha=1` recorded the strongest observed test RMSE and R², but it was not chosen post hoc from test results.
- The selected model reduced MAE by approximately 28% versus the mean baseline.
- Extreme high-revenue customers remain difficult to predict, with residual evidence of heteroscedasticity.

## Business Interpretation

With further validation and appropriate operational design, a model of this kind could potentially help a telecom operator:

- Estimate customer value and support revenue planning.
- Segment or prioritize customers for retention and service strategies.
- Identify likely high-value customers for targeted engagement.
- Investigate customers whose observed revenue differs substantially from expected revenue.

The current analysis is exploratory and should not be considered a deployed decision system.

## Limitations

- The analysis uses a relatively small dataset (2,000 observations).
- The target is highly right-skewed and includes extreme high-revenue observations.
- Residual diagnostics indicate heteroscedasticity.
- The model has difficulty with extreme high-revenue customers.
- Leakage prevention intentionally restricts the feature set by excluding revenue-derived variables.
- Results are specific to this dataset and held-out split.

## Future Improvements

- Compare robust, non-linear, and tree-based regression approaches.
- Engineer additional non-revenue telecom behavior features.
- Investigate the behavior of extreme high-revenue observations and strategies for modeling them.
- Use more extensive validation, such as repeated or nested cross-validation.
- Validate on additional or external telecom data before considering operational use.

## Project Structure

```text
telecom-regression/
├── app.py
├── data/
│   ├── processed/
│   │   └── telecom_regression_model_data.csv
│   └── raw/
│       └── South Asian Wireless Telecom Operator (SATO 2015).csv
├── images/
│   ├── Boxplot of Aggregate Total Revenue.png
│   ├── Correlation Matrix — Regression Variables.png
│   ├── Distribution of Aggregate Total Revenue.png
│   ├── Residuals vs Predicted Revenue.png
│   └── Ridge Regression- Residuals vs Predicted Revenue.png
├── models/
│   └── ridge_regression_alpha100.joblib
├── notebooks/
│   ├── 01_regression_eda.ipynb
│   ├── 02_regression_modeling.ipynb
│   └── load_csv.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

The workflow is separated into two notebooks:

- `notebooks/01_regression_eda.ipynb` loads and understands the raw data, investigates leakage, performs EDA and data-quality checks, cleans and prepares `model_df`, and saves the prepared 2,000 × 9 modeling dataset to `data/processed/telecom_regression_model_data.csv`.
- `notebooks/02_regression_modeling.ipynb` loads the prepared data, performs the train/test split, and runs the baseline, Linear Regression, Ridge and Lasso experiments, cross-validation, diagnostics, coefficient interpretation, and log-target experiment. It exports the CV-selected Ridge pipeline (`alpha=100`) as `models/ridge_regression_alpha100.joblib`.

`notebooks/load_csv.ipynb` is a small dataset-acquisition helper that downloads the Kaggle dataset with `kagglehub` and copies the CSV into `data/raw/`.

## Technologies Used

- Python
- Pandas and NumPy
- scikit-learn
- Matplotlib and Seaborn
- Jupyter Notebook
- `kagglehub` (dataset-acquisition helper notebook)
- Streamlit (local prediction application)
- Joblib (saved model loading)

## Streamlit Deployment

The project includes a local Streamlit application in `app.py` for generating customer revenue estimates with the selected Ridge Regression pipeline (`alpha=100`). The app loads `models/ridge_regression_alpha100.joblib` with Joblib and predicts `Aggregate_Total_Rev`.

The interface accepts the eight model features:

- `network_age`
- `Aggregate_Data_Vol`
- `Aggregate_Calls`
- `Aggregate_complaint_count`
- `aug_user_type`
- `sep_user_type`
- `aug_fav_a`
- `sep_fav_a`

The saved pipeline contains the fitted preprocessing steps as well as the Ridge estimator, so the app can pass raw numerical and categorical inputs directly to the trained pipeline. It displays the predicted revenue together with the model's held-out test metrics—R² **0.198** and MAE **487.13**—and warns that an individual prediction is an estimate that may differ substantially from actual customer revenue.

The application is currently intended for local use only; no public Streamlit deployment is claimed.

### Local Deployment Check

As a local input/output check, the app was tested with one real SATO 2015 observation. It returned a prediction of **1053.24** for an actual `Aggregate_Total_Rev` of **1592.72**. This is a single illustrative deployment test, not a model-performance metric; the held-out test metrics reported above remain the appropriate performance summary.

## How to Run

1. Clone the repository and create a virtual environment (Python 3.11 or a compatible Python version is recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. Install the packages used by the notebooks:

   ```bash
   pip install -r requirements.txt
   ```

3. Obtain the dataset from [Kaggle](https://www.kaggle.com/datasets/mahreen/sato2015) and place `South Asian Wireless Telecom Operator (SATO 2015).csv` in `data/raw/`. Alternatively, run `notebooks/load_csv.ipynb`, which uses `kagglehub` to download and copy the file.

4. Start Jupyter and run the analysis notebooks in order. Run all cells in `01_regression_eda.ipynb` first; it creates the processed dataset consumed by `02_regression_modeling.ipynb`:

   ```bash
   jupyter notebook
   ```

5. To run the local prediction application, ensure that `models/ridge_regression_alpha100.joblib` is present and run:

   ```bash
   streamlit run app.py
   ```

## Repository Hygiene

- `.gitignore` excludes OS metadata, virtual environments, Python/Jupyter generated files, locally downloaded CSV files, and generic exported model formats. It explicitly retains `models/ridge_regression_alpha100.joblib` for the local Streamlit application.
- `requirements.txt` lists the notebook and application dependencies, including Streamlit and Joblib.
- The raw Kaggle CSV remains locally available for notebook execution but is ignored by Git. Obtain it from Kaggle or with `load_csv.ipynb` after cloning.
- `models/ridge_regression_alpha100.joblib` is the exported CV-selected Ridge pipeline used by `app.py`.

## Author

Altayeb Alshiekh  
GitHub: [Altayeb5](https://github.com/Altayeb5)
