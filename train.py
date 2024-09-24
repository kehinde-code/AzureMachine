# train.py
from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Load workspace
ws = Workspace.from_config()

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
os.makedirs("outputs", exist_ok=True)
joblib.dump(model, "outputs/model.joblib")

# Log metrics
accuracy = model.score(X_test, y_test)
run = ws.get_default_experiment().start_logging()
run.log("accuracy", accuracy)
run.complete()
