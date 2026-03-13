# Financial Fraud Detection – Feature Engineering

## Overview

This module performs **feature engineering** on the cleaned transaction dataset to enhance the predictive capability of machine learning models used for financial fraud detection.

The script transforms the raw transactional dataset into a richer dataset by deriving behavioral, temporal, statistical, and risk-based features.

The final engineered dataset contains **101,322 rows and 34+ features**, which are used for downstream machine learning tasks such as fraud classification and anomaly detection.

---

# Input Dataset

The feature engineering pipeline takes the following file as input:

```
cleaned_dataset.csv
```

This dataset contains transaction-level information including:

* Date and Time of transaction
* Sender and receiver accounts
* Transaction amount
* Payment currency
* Sender and receiver bank locations
* Transaction type
* Laundering label

---

# Output Dataset

The script generates the following file:

```
engineered_dataset.csv
```

This dataset includes the original variables along with newly derived features that capture behavioral patterns and potential fraud indicators.

---

# Feature Engineering Process

## Datetime Processing

Date and Time columns are combined to create a unified timestamp:

```
Datetime = Date + Time
```

This enables time-based analysis such as hourly patterns and rolling transaction velocity.

---

# Engineered Features

## Transaction Behavior Features

### is_cross_border

Indicates whether the sender and receiver banks are located in different countries.

```
1 → Cross-border transaction  
0 → Domestic transaction
```

### currency_mismatch

Identifies transactions where the payment currency differs from the received currency.

### transaction_hour

Extracts the hour (0–23) from the transaction timestamp.

### day_of_week

Represents the day of the week:

| Value | Day    |
| ----- | ------ |
| 0     | Monday |
| 6     | Sunday |

---

# Sender and Receiver Activity Metrics

### sender_txn_count

Total number of transactions initiated by the sender.

### receiver_txn_count

Total number of transactions received by the receiver.

### sender_receiver_txn_count

Total number of transactions between a specific sender–receiver pair.

---

# Transaction Amount Analysis

### sender_avg_amount

Average amount historically sent by the sender.

### amount_deviation

Difference between the transaction amount and the sender’s historical average.

### amount_deviation_ratio

Ratio between the current transaction amount and the sender's average transaction value.

---

# Country Risk Score

Risk levels are assigned based on the **frequency of transactions from each country**.

| Frequency        | Risk Score      |
| ---------------- | --------------- |
| Rare (<5%)       | High Risk (3)   |
| Moderate (5–20%) | Medium Risk (2) |
| Common (>20%)    | Low Risk (1)    |

Two scores are computed:

* Sender country risk
* Receiver country risk

These are combined to produce:

```
country_risk_score
```

---

# Transaction Velocity

### transaction_velocity_24h

Rolling count of transactions performed by the same sender within the last **24 hours**.

High velocity may indicate:

* automated attacks
* account takeover
* layering in money laundering.

---

# Currency Risk Feature

### currency_risk

Currencies are assigned risk scores based on how frequently they appear in the dataset.

Rare currencies are treated as **higher risk**.

---

# Fraud Risk Flags

## high_amount_flag

Activated when a transaction amount exceeds **three times the sender's average transaction amount**.

## new_receiver

Flags the first transaction between a sender and a receiver.

## night_transaction

Flags transactions occurring between **12:00 AM and 5:00 AM**.

## velocity_flag

Triggered when a sender performs **more than 10 transactions within 24 hours**.

---

# Additional Behavioral Features

### receiver_avg_amount

Average amount typically received by a specific receiver.

### amount_to_receiver_avg_ratio

Compares current amount with the receiver's historical average.

### is_weekend

Indicates whether the transaction occurred on a weekend.

```
1 → Saturday or Sunday
0 → Weekday
```

### time_since_last_txn_seconds

Time difference (in seconds) since the sender’s previous transaction.

Very small intervals may indicate **automated transaction bursts**.

---

# Advanced Features

## unique_receivers_per_sender

Counts how many **different receivers** a sender has transacted with.

Useful for detecting **1-to-many laundering patterns**.

---

## unique_senders_per_receiver

Counts the number of **different senders** transferring funds to a receiver.

Useful for detecting **smurfing and mule accounts**.

---

## sender_country_risk

Calculated using **target encoding** based on the historical laundering rate for each sender country.

Higher values indicate regions historically associated with more laundering cases.

---

## receiver_country_risk

Similar to sender_country_risk but applied to the receiver bank location.

---

# Short-Term Transaction Velocity

## sender_transactions_last_1_hour

Number of transactions performed by the same sender within the last **1 hour**.

Detects rapid bursts of activity.

---

## global_transactions_last_1_hour

Total number of transactions across all users within the last **1 hour**.

Helps identify abnormal global spikes.

---

# Final Dataset Summary

| Metric      | Value                  |
| ----------- | ---------------------- |
| Rows        | 101,322                |
| Columns     | 34+                    |
| Input File  | cleaned_dataset.csv    |
| Output File | engineered_dataset.csv |

---

# Usage

Run the feature engineering script:

```
python feature_engineering.py
```

The script will generate the engineered dataset automatically.

---

# Purpose

The engineered features are designed to improve the performance of machine learning models such as:

* Logistic Regression
* Random Forest
* XGBoosting
* Isolation Forest

These models will use the enriched dataset to detect suspicious financial transactions and potential money laundering activities.

---
