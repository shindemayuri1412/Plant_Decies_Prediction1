import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
data = pd.read_csv("growth_data.csv")

# Display first rows
print(data.head())

# Encode categorical columns
le_crop = LabelEncoder()
le_soil = LabelEncoder()
le_stage = LabelEncoder()
le_result = LabelEncoder()

data['crop_enc'] = le_crop.fit_transform(data['crop ID'])
data['soil_enc'] = le_soil.fit_transform(data['soil_type'])
data['stage_enc'] = le_stage.fit_transform(data['Seedling Stage'])
data['result_enc'] = le_result.fit_transform(data['result'])

# Features and target
X = data[['crop_enc', 'soil_enc', 'stage_enc', 'MOI', 'temp', 'humidity']]
y = data['result_enc']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Convert target names to strings to avoid TypeError
target_names = [str(c) for c in le_result.classes_]

print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

# Save the model and encoders
joblib.dump(clf, "model2_disease_predictor.pkl")
joblib.dump(le_crop, "le_crop.pkl")
joblib.dump(le_soil, "le_soil.pkl")
joblib.dump(le_stage, "le_stage.pkl")
joblib.dump(le_result, "le_result.pkl")

print("Model and encoders saved successfully!")
