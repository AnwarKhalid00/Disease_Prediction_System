import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


#  Load Training and Testing Data

train_data = pd.read_csv("dataset/Training.csv")
test_data = pd.read_csv("dataset/Testing.csv")

print("Training Data Shape:", train_data.shape)
print("Testing Data Shape:", test_data.shape)

# Remove unwanted columns

train_data = train_data.loc[:, ~train_data.columns.str.contains("^Unnamed")]
test_data = test_data.loc[:, ~test_data.columns.str.contains("^Unnamed")]

# Convert symptom columns to numeric

for col in train_data.columns:
    if col != "prognosis":
        train_data[col] = pd.to_numeric(train_data[col], errors="coerce").fillna(0).astype("int8")

for col in test_data.columns:
    if col != "prognosis":
        test_data[col] = pd.to_numeric(test_data[col], errors="coerce").fillna(0).astype("int8")

# Add Lifestyle Features (Simulated)

# Training data lifestyle features
train_data["smoking"] = np.random.randint(0, 2, size=len(train_data))
train_data["alcohol"] = np.random.randint(0, 2, size=len(train_data))
train_data["exercise_level"] = np.random.randint(0, 4, size=len(train_data))
train_data["water_intake"] = np.random.randint(1, 6, size=len(train_data))
train_data["bmi"] = np.random.randint(18, 35, size=len(train_data))

# Testing data lifestyle features
test_data["smoking"] = np.random.randint(0, 2, size=len(test_data))
test_data["alcohol"] = np.random.randint(0, 2, size=len(test_data))
test_data["exercise_level"] = np.random.randint(0, 4, size=len(test_data))
test_data["water_intake"] = np.random.randint(1, 6, size=len(test_data))
test_data["bmi"] = np.random.randint(18, 35, size=len(test_data))

# Split Features and Target

X_train = train_data.drop("prognosis", axis=1)
y_train = train_data["prognosis"]

X_test = test_data.drop("prognosis", axis=1)
y_test = test_data["prognosis"]

# Aligning the columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Train RandomForest Model

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    n_jobs=-1,
    class_weight="balanced",
    random_state=42
)

print("\nTraining model...")
model.fit(X_train, y_train)

# Training Accuracy

train_accuracy = accuracy_score(y_train, model.predict(X_train))

# Testing Accuracy

test_accuracy = accuracy_score(y_test, model.predict(X_test))

# Printing the training and testing output

print("Training Accuracy:", round(train_accuracy * 100, 2), "%")
print("Testing Accuracy:", round(test_accuracy * 100, 2), "%")

# Saving the Model

pickle.dump(model, open("model/disease_model.pkl", "wb"))
pickle.dump(X_train.columns, open("model/columns.pkl", "wb"))

print("\nModel saved successfully!")