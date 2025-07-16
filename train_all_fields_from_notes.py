import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.utils.multiclass import unique_labels

# Load the dataset
# Ensure the path is correct; update if necessary
csv_path = "consolidated_earthquake_observations_20250703.csv"  # Update if file is renamed or moved
df = pd.read_csv(csv_path)

# Set up directory for models
os.makedirs("models", exist_ok=True)
df = df[df["Notes"].notnull() & (df["Notes"].str.strip() != "")]
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(df["Notes"])
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

# Define structured fields to model 
excluded = ["Notes", "OBJECTID", "GlobalID", "CreationDate", "EditDate", "Creator", "Editor"]
structured_fields = [col for col in df.columns if col not in excluded and not col.startswith("_")]

# Model loop
results = []
for field in structured_fields:
    print(f"\n🔍 Training model for: {field}")
    y_raw = df[field]

    # Skip if all null or only 1 value
    if y_raw.isnull().all() or y_raw.nunique() < 2:
        print(f"⚠️ Skipping {field} (insufficient data)")
        continue

    # Filter valid rows for this field
    mask = y_raw.notnull()
    X = X_tfidf[mask.to_numpy()]
    y = y_raw[mask]

    try:
        if y.dtype == "object" or y.nunique() < 25:
            # Classification
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            print(f"✅ {field} accuracy: {acc:.2f}")
            print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

            joblib.dump(clf, f"models/{field}_classifier.pkl")
            joblib.dump(label_encoder, f"models/{field}_label_encoder.pkl")
            results.append((field, "classification", acc))

        else:
            # Regression
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            reg = RandomForestRegressor(n_estimators=100, random_state=42)
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            print(f"✅ {field} R² score: {r2:.2f}")

            joblib.dump(reg, f"models/{field}_regressor.pkl")
            results.append((field, "regression", r2))

    except Exception as e:
        print(f"❌ Error training {field}: {str(e)}")
        results.append((field, "error", str(e)))

# Summary 
results_df = pd.DataFrame(results, columns=["Field", "Type", "Score"])
results_df.to_csv("models/training_summary.csv", index=False)
print("\n✅ All models trained. Summary saved to models/training_summary.csv")
