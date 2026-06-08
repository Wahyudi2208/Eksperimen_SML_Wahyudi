import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
import os
dagshub.init(repo_owner='Wahyudi2208', repo_name='Telco-Churn-MLflow', mlflow=True)

mlflow.set_experiment("Telco_Churn_Tuning")

df = pd.read_csv("telco_preprocessed.csv")
X = df.drop(columns=["Churn Value"])
y = df["Churn Value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced"),
    param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

with mlflow.start_run(run_name="RF_Tuned_Skilled"):

    mlflow.log_params(grid.best_params_)

    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    mlflow.log_metric("accuracy",  accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_score",  f1_score(y_test, y_pred))
    mlflow.log_metric("roc_auc",   roc_auc_score(y_test, y_proba))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall",    recall_score(y_test, y_pred))

    mlflow.sklearn.log_model(best_model, "model")

    # Artefak 1: Confusion Matrix
    os.makedirs("artifacts", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Not Churn", "Churn"]).plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix — RF Tuned")
    fig.savefig("artifacts/confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact("artifacts/confusion_matrix.png")

    # Artefak 2: Feature Importance
    import pandas as pd
    feat_df = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    feat_df.plot(kind="barh", ax=ax2, color="steelblue")
    ax2.invert_yaxis()
    ax2.set_title("Top 15 Feature Importance")
    fig2.savefig("artifacts/feature_importance.png", bbox_inches="tight")
    plt.close(fig2)
    mlflow.log_artifact("artifacts/feature_importance.png")

    report = classification_report(y_test, y_pred, target_names=["Not Churn", "Churn"])
    mlflow.log_text(report, "classification_report.txt")

    print(f"Best Params: {grid.best_params_}")
    print(f"F1 Score   : {f1_score(y_test, y_pred):.4f}")

print("Selesai!")