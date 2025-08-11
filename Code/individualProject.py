import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

# Simulate dataset
np.random.seed(0)
data_size = 395
df = pd.DataFrame({
    'studytime': np.random.choice([1, 2, 3, 4], data_size),
    'failures': np.random.poisson(1.0, data_size),
    'internet': np.random.choice([0, 1], data_size),
    'famsup': np.random.choice([0, 1], data_size),
    'Dalc': np.random.randint(1, 5, data_size),
    'G3': np.clip(np.random.normal(11, 3, data_size).astype(int), 0, 20)
})
df['pass'] = (df['G3'] >= 10).astype(int)

# Correlation heatmap
corr_matrix = df[['studytime', 'failures', 'internet', 'famsup', 'Dalc', 'G3']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Fig. 1: Correlation Matrix of Key Features")
plt.tight_layout()
plt.savefig("fig1_correlation_matrix.png")
plt.close()

# Logistic Regression
features = ['studytime', 'failures', 'internet', 'famsup', 'Dalc']
X = df[features]
y = df['pass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

# Confusion matrix for logistic regression
cm = confusion_matrix(y_test, y_pred_log)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Fig. 2: Confusion Matrix (Logistic Regression)")
plt.tight_layout()
plt.savefig("fig2_confusion_matrix.png")
plt.close()

# ROC curve for logistic regression
fpr, tpr, _ = roc_curve(y_test, y_prob_log)
roc_auc = roc_auc_score(y_test, y_prob_log)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Fig. 3: ROC Curve (Logistic Regression)")
plt.legend()
plt.tight_layout()
plt.savefig("fig3_roc_curve.png")
plt.close()



