# ConTextTab: A Semantics-Aware Tabular In-Context Learner
[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/contexttab)](https://api.reuse.software/info/github.com/SAP-samples/contexttab)

## Description

Implementation of the deep learning model with the inference pipeline described in the paper "ConTextTab: A Semantics-Aware Tabular In-Context Learner". Link to the paper: [ARXIV LINK]

![logo](https://github.com/SAP-samples/contexttab/blob/main/ConTextTab_architecture.png)

## Abstract

Tabular in-context learning (ICL) has recently achieved state-of-the-art (SOTA) performance on several tabular prediction tasks. Previously restricted to classification problems on small tables, recent advances such as TabPFN and TabICL have extended its use to larger datasets. While being architecturally efficient and well-adapted to tabular data structures, current table-native ICL architectures, being trained exclusively on synthetic data, do not fully leverage the rich semantics and world knowledge contained in real-world tabular data. On another end of this spectrum, tabular ICL models based on pretrained large language models such as TabuLa-8B integrate deep semantic understanding and world knowledge but are only able to make use of a small amount of context due to inherent architectural limitations. With the aim to combine the best of both these worlds, we introduce **ConTextTab**, integrating semantic understanding and alignment into a table-native ICL framework. By employing specialized embeddings for different data modalities and by training on large-scale real-world tabular data, our model is competitive with SOTA across a broad set of benchmarks while setting a new standard on the semantically rich CARTE benchmark.

## Requirements

The requirements are detailed in the `requirements.txt` file

## Basic Usage

### Classification

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from contexttab.contexttab import ConTextTabClassifier
from contexttab.constants import ModelSize

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize a classifier
clf = ConTextTabClassifier(ModelSize.base,
                           './contexttab/checkpoints/0.1_l2/base.pt',
                           bagging=1,
                           max_context_size=2048,
                           regression_type='l2',
                           classification_type='cross-entropy',
                           is_load_rnn=True)

clf.fit(X_train, y_train)

# Predict probabilities
prediction_probabilities = clf.predict_proba(X_test)
print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))

# Predict labels
predictions = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, predictions))
```

### Regression
```python
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from contexttab.contexttab import ConTextTabRegressor
from contexttab.constants import ModelSize

# Load Boston Housing data
df = fetch_openml(data_id=531, as_frame=True)  # Boston Housing dataset
X = df.data
y = df.target.astype(float)  # Ensure target is float for regression

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize the regressor
regressor = ConTextTabRegressor(ModelSize.base,
                                './contexttab/checkpoints/0.1_l2/base.pt',
                                bagging=1,
                                max_context_size=2048,
                                regression_type='l2',
                                classification_type='cross-entropy',
                                is_load_rnn=True)

regressor.fit(X_train, y_train)

# Predict on the test set
predictions = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)
```

## Citations
If you use this dataset in your research or want to refer to our work, please cite: [TODO]

## Known Issues
No known issues

## How to obtain support
[Create an issue](https://github.com/SAP-samples/contexttab/issues) in this repository if you find a bug or have questions about the content.

## Contributing
If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License
Copyright (c) 2024 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSE) file.
