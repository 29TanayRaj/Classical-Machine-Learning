# Train-Test Split (Detailed Notes)

## What is Train-Test Split?

**Train-Test Split** is a fundamental concept in machine learning where
the available dataset is divided into two separate parts:

1.  **Training Set** -- Used to train the machine learning model by
    allowing it to learn patterns, relationships, and trends from the
    data.
2.  **Testing Set** -- Used to evaluate how well the trained model
    performs on data it has never seen before.

The main objective of a machine learning model is **not just to memorize
the training data**, but to **generalize well** and make accurate
predictions on new, unseen data. The train-test split helps us measure
this ability.


# Why Do We Need a Train-Test Split?

If we train and evaluate a model on the same data, the model may simply
memorize the dataset instead of learning meaningful patterns. This leads
to **overfitting**, where the model performs extremely well on the
training data but poorly on new data.

By keeping a separate testing dataset, we can estimate how the model is
likely to perform in real-world situations.

**In simple words:**

-   Training Data → Used for learning.
-   Testing Data → Used for checking whether the learning is effective.


# Common Train-Test Split Ratios

  Training   Testing   When Used
  ---------- --------- -------------------------
  80%        20%       Standard recommendation
  70%        30%       More testing data
  90%        10%       Very large datasets
  75%        25%       Common alternative

An **80/20 split** provides a good balance between learning and
evaluation.


# Similar Data Distribution

The training and testing datasets should have **similar distributions**
so that both represent the original dataset fairly.

For **classification problems**, use a **Stratified Train-Test Split**,
which preserves the class proportions in both datasets.


# Random Shuffling

For most machine learning datasets, shuffle the data before splitting to
ensure both datasets contain a representative mix of samples.

**Exception:** Do **not** shuffle time series data.


# Workflow

``` text
Complete Dataset
        │
        ▼
Random Shuffle
        │
        ▼
Train-Test Split
   │             │
   ▼             ▼
Training Set   Testing Set
      │
      ▼
Train Model
      │
      ▼
Make Predictions
      │
      ▼
Evaluate on Testing Set
```

# Advantages

-   Easy to implement.
-   Evaluates performance on unseen data.
-   Detects overfitting and underfitting.
-   Estimates real-world performance.


# Limitations

-   Results depend on the quality of the split.
-   Small datasets may produce unreliable estimates.
-   Different random splits can give different results.

# Best Practices

-   Use an **80/20** split as a starting point.
-   Shuffle the data before splitting (except for time series).
-   Keep similar distributions in both datasets.
-   Use **stratified splitting** for classification.
-   Never use testing data during training.


# Train-Test Split for Time Series Data

## Why is Time Series Different?

Time series data consists of observations collected over time, such as
stock prices, sales, weather, or sensor readings.

The chronological order is important because future values depend on
past values.

## Do Not Shuffle Time Series Data

Random shuffling introduces **data leakage**, where the model gains
access to future information during training.

**Incorrect:**

``` text
Jan → Feb → Mar → Apr → May → Jun

Random Shuffle

Training: Mar Jan Jun
Testing : Feb Apr May
```

## Correct Chronological Split

Always split by time.

``` text
Jan → Feb → Mar → Apr → May → Jun

Training: Jan → Feb → Mar → Apr
Testing : May → Jun
```

The model learns from the past and predicts the future.


## Example

Monthly sales from January 2022 to December 2024:

-   Training: January 2022 -- September 2024
-   Testing: October 2024 -- December 2024

## Time Series Cross-Validation

Instead of standard K-Fold, use **Time Series (Walk-Forward)
Cross-Validation**.

``` text
Split 1:
Train: Jan Feb Mar
Test : Apr

Split 2:
Train: Jan Feb Mar Apr
Test : May

Split 3:
Train: Jan Feb Mar Apr May
Test : Jun
```

## Best Practices for Time Series

-   Never shuffle the data.
-   Split chronologically.
-   Testing data must come after training data.
-   Prevent data leakage.
-   Use Time Series Cross-Validation.

## Regular Data vs Time Series

  Feature           Regular ML Data   Time Series Data
  ----------------- ----------------- ------------------
  Shuffle           Yes               No
  Order Important   No                Yes
  Split             Random            Chronological
  Validation        K-Fold            Walk-Forward

------------------------------------------------------------------------

# Summary

For most machine learning problems, randomly shuffle the data and use an
**80/20 train-test split**. For **time series data**, never shuffle;
always split chronologically so that the model learns from historical
data and is evaluated on future data, preventing data leakage and
producing realistic performance estimates.
