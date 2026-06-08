import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("telco_preprocessed.csv")
X = df.drop(columns=["Churn Value"])
y = df["Churn Value"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# MLflow setup
mlflow.set_experiment("Telco_Churn_Experiment")

with mlflow.start_run(run_name="RandomForest_Baseline"):
    # Autolog aktifkan SEBELUM fit
    mlflow.sklearn.autolog()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"   # handle imbalance churn
    )
    model.fit(X_train, y_train)

    # Evaluasi manual (opsional, autolog sudah log metrics)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

print("Training selesai. Jalankan: mlflow ui")