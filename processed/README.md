
# Financial Transaction Fraud Detection – Data Preprocessing

## Project Overview

This project focuses on **preprocessing financial transaction data** to prepare it for machine learning models used in **money laundering detection**. Financial datasets often contain inconsistencies, extreme values, and categorical information that cannot be directly used in machine learning algorithms.

The objective of this preprocessing pipeline is to clean, transform, and encode the dataset so that it becomes **structured, reliable, and suitable for fraud detection models such as Random Forest, XGBoost, etc**.

The dataset used in this stage contains **101,322 financial transactions** with **12 attributes** including account information, transaction amount, currency types, bank locations, and laundering labels.

---

# Dataset Description

The dataset contains the following features:

| Column                 | Description                                  |
| ---------------------- | -------------------------------------------- |
| Time                   | Time of the transaction                      |
| Date                   | Date of the transaction                      |
| Sender_account         | Unique ID of the sender                      |
| Receiver_account       | Unique ID of the receiver                    |
| Amount                 | Transaction amount                           |
| Payment_currency       | Currency used for payment                    |
| Received_currency      | Currency received                            |
| Sender_bank_location   | Sender's bank location                       |
| Receiver_bank_location | Receiver's bank location                     |
| Payment_type           | Method of payment                            |
| Is_laundering          | Target variable (1 = Laundering, 0 = Normal) |
| Laundering_type        | Type/category of laundering                  |

Dataset size:

* **Rows:** 101,322
* **Columns:** 12

---

# Initial Data Exploration

Basic exploratory analysis was performed to understand the dataset.

Key observations:

* No **missing values** were found in any column.
* No **duplicate rows** were detected.
* Dataset contains **8 categorical columns** and **4 numerical columns**.
* The **Amount column contains large outliers**, which required treatment.

---

# Data Preprocessing Steps

The following preprocessing steps were implemented:

## 1. Loading the Dataset

The dataset was loaded using **Pandas**.

```python
df = pd.read_csv('reduced.csv')
```

Initial exploration included:

* Data type inspection
* Missing value analysis
* Duplicate detection
* Summary statistics

---

# 2. Handling Missing Values

Although the dataset did not contain missing values, a **generic preprocessing step** was included to ensure robustness.

```python
df = df.dropna()
```

This guarantees that future datasets with missing values will be handled automatically.

---

# 3. Removing Duplicate Records

Duplicate rows can bias machine learning models and distort statistical analysis.

```python
df = df.drop_duplicates()
```

In this dataset:

* **0 duplicate rows were found**
* The step was retained to maintain pipeline consistency.

---

# 4. Outlier Treatment

The **Amount column** showed extreme values compared to its distribution.

To prevent models from being influenced by extreme transactions, **percentile clipping** was applied.

```python
q_low = df["Amount"].quantile(0.01)
q_hi = df["Amount"].quantile(0.99)

df["Amount"] = df["Amount"].clip(lower=q_low, upper=q_hi)
```

Benefits of clipping:

* Preserves all rows
* Prevents extreme values from skewing the model
* Maintains statistical integrity

---

# 5. Encoding Categorical Variables

Machine learning models require **numerical inputs**, so categorical variables were converted into numeric form using **Label Encoding**.

Columns encoded:

* Payment_currency
* Received_currency
* Sender_bank_location
* Receiver_bank_location
* Payment_type
* Laundering_type

Example code:

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))
```

This transformation converts categorical values into integer labels that models can interpret.

---

# 6. Date and Time Columns

The dataset contains **Date** and **Time** columns.

These were preserved in their original format because they can later be used for:

* Temporal fraud detection
* Transaction pattern analysis
* Time-based feature engineering

---

# Output Dataset

After preprocessing, the cleaned dataset was exported.

```python
df.to_csv('cleaned_dataset.csv', index=False)
```

Output file:

```
cleaned_dataset.csv
```

This dataset is now ready for:

* Machine learning modeling
* Fraud detection algorithms
* Feature engineering
* Data visualization

---

# Final Preprocessed Dataset Characteristics

| Feature               | Status                        |
| --------------------- | ----------------------------- |
| Missing Values        | Removed                       |
| Duplicate Records     | Removed                       |
| Outliers              | Clipped (1st–99th percentile) |
| Categorical Variables | Label Encoded                 |
| Dataset Ready for ML  | Yes                           |

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook

---

# Next Steps

After preprocessing, the following steps will be performed:

* Feature Engineering
* Data Balancing (SMOTE / Undersampling)
* Model Training
* Fraud Detection Model Evaluation
* Performance Metrics Analysis

Potential models include:

* Random Forest
* XGBoost
* Logistic Regression



