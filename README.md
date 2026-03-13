## Financial Fraud Detection – Cohort 14

This repository contains an **end‑to‑end financial transaction fraud detection pipeline**.  
Starting from raw transactional data, it walks through:

- **Data cleaning & preprocessing**
- **Feature engineering**
- **Standardization / normalization**
- **Train–test splitting**
- **Modeling with tree‑based and anomaly‑detection algorithms**
- **Evaluation using precision–recall and AUPRC**

The goal is to build robust models that can help identify **potential money laundering and fraudulent transactions**.

---

## Repository Structure

- `data/`  
  Raw and reduced versions of the original transaction dataset (e.g., `raw.csv`, `reduced_dataset.csv`).

- `processed/`  
  Notebooks and scripts for **data cleaning and preprocessing**.  
  Output: `cleaned_dataset.csv`  
  See `processed/README.md` for full details.

- `feature_engineering/`  
  Notebooks and scripts that create **engineered features** (behavioral, temporal, risk‑based, velocity features, etc.).  
  Output: `engineered_dataset.csv`  
  See `feature_engineering/README.md` for a detailed description.

- `Std&Nor/`  
  Code for **standardizing and normalizing** numerical features (e.g., `std&nor.py`, `normalized_dataset.csv`).

- `XGBoost/`  
  Notebooks and data for **gradient‑boosted tree models** (e.g., `XGboostmodel.ipynb`, `feature_engineered.csv`).

- `Isolation_Forest.ipynb`  
  Notebook exploring **unsupervised anomaly detection** using Isolation Forest on transaction data.

- `Hyperparameter_optimization.ipynb`  
  Notebook for **tuning model hyperparameters** (e.g., XGBoost / tree‑based models).

- `cross border.ipynb`  
  Analysis focused on **cross‑border transaction behavior and risk**.

- `Strategic Train & Test Split of Financial Fraud Detection .py`  
  Script for **strategic train–test splitting** that respects temporal order / leakage concerns.

- `LICENSE`  
  Project license information.

---

## Data Pipeline Overview

1. **Raw Data → Processed Data**
   - Load raw CSVs from `data/`
   - Handle missing values and duplicates
   - Clip extreme outliers in `Amount`
   - Encode categorical features (e.g., currencies, locations, payment types)  
   - Export: `processed/cleaned_dataset.csv`

2. **Processed Data → Engineered Features**
   - Combine `Date` and `Time` into a timestamp
   - Create behavioral and temporal features (hour of day, day of week, weekend flag, night transactions)
   - Compute sender/receiver statistics (transaction counts, averages, deviations, velocity)
   - Assign country and currency risk scores
   - Build fraud‑risk flags (high amount, new receiver, high velocity, etc.)  
   - Export: `feature_engineering/engineered_dataset.csv`

3. **Feature Scaling**
   - Apply standardization / normalization to numerical columns using scripts in `Std&Nor/`
   - Export a model‑ready normalized dataset (e.g., `normalized_dataset.csv`)

4. **Modeling & Evaluation**
   - **Supervised models** (e.g., Random Forest, XGBoost, Logistic Regression) using feature‑engineered and scaled data
   - **Unsupervised models** (e.g., Isolation Forest) for anomaly detection
   - Evaluate performance with:
     - Precision, Recall, F1‑score
     - ROC‑AUC (if relevant)
     - **Precision–Recall curves and AUPRC**

---

## Getting Started

### 1. Environment Setup

1. Install Python 3.9+ (recommended).  
2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows
```

3. Install required packages (example – adapt to your environment):

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

If you already have a `requirements.txt`, you can instead run:

```bash
pip install -r requirements.txt
```

---

### 2. Running the Pipeline (Typical Order)

1. **Preprocessing**  
   - Open the notebook in `processed/` (or run the associated script) to generate `cleaned_dataset.csv`.

2. **Feature Engineering**  
   - Use the notebook / script in `feature_engineering/` to create `engineered_dataset.csv`.

3. **Scaling / Normalization**  
   - Run `Std&Nor/std&nor.py` (or the relevant notebook) to produce `normalized_dataset.csv`.

4. **Model Training**  
   - Open:
     - `XGBoost/XGboostmodel.ipynb` for gradient‑boosted models, and/or
     - `Isolation_Forest.ipynb` for anomaly detection.

5. **Evaluation & Visualization**  
   - Use the precision‑recall / AUPRC notebooks or scripts to compare models and analyze thresholds.

---

## Dataset Notes

- The dataset consists of **financial transactions** with fields such as:
  - Sender and receiver account IDs
  - Amount, currencies used
  - Bank locations
  - Payment type
  - Temporal information (date, time)
  - Laundering / fraud labels
- Class imbalance is expected (fraud is rare); consider:
  - **SMOTE**, undersampling, or class‑weighted models.

---

## Intended Audience

This project is suitable for:

- Students and practitioners learning **fraud detection and ML for tabular data**
- Data scientists experimenting with **feature engineering and imbalance handling**
- Teams prototyping **transaction monitoring** and **AML (Anti‑Money Laundering)** use cases

Feel free to extend the notebooks, add new models, or integrate the pipeline into a production‑grade API.
