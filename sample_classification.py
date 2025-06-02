
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
