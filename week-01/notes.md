# Week-01 Notes: Supervised Learning

## What is Supervised Learning?

Supervised learning is a category of machine learning where the model learns from a dataset that includes both input features and the correct output labels. During training, the algorithm adjusts its internal parameters to minimize the difference between its predictions and the actual labels.

**Core idea:** Given a set of labeled examples, build a function that maps inputs to outputs, then use that function to predict labels for new, unseen data.

## Key Terminology

| Term | Description |
|------|-------------|
| **Feature (X)** | The input variables used to make a prediction |
| **Label (y)** | The target variable the model tries to predict |
| **Training set** | The portion of data used to fit the model |
| **Test set** | The held-out portion used to evaluate performance |

## Logistic Regression

Despite the name, logistic regression is a **classification** algorithm, not a regression one. It estimates the probability that a given input belongs to a particular class by applying the sigmoid function to a linear combination of the features:

```
P(y=1 | X) = 1 / (1 + e^(-(w·X + b)))
```

If the probability exceeds 0.5, the sample is classified as class 1; otherwise it is class 0.

## This Week's Problem

We predict whether a student passes or fails an exam based on:
- Daily study hours
- Lecture attendance percentage

**Label mapping:** `0` → Fail, `1` → Pass

## Libraries Used

```
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
```

Install with:
```bash
pip install numpy scikit-learn matplotlib
```

## How to Run

```bash
python supervised_learning.py
```

The script will print accuracy metrics and save a decision-boundary plot as `sinav_tahmin_grafik.png`.
