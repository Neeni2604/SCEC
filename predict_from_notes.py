import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# === Load the TF-IDF vectorizer ===
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# === List of fields to predict ===
# These are based on your saved model filenames
model_files = os.listdir("models")
fields = sorted(set("_".join(f.split("_")[:-1]) for f in model_files if f.endswith(".pkl") and f != "tfidf_vectorizer.pkl"))
print(f"Loaded model fields: {fields}")


# === Define prediction function ===
def predict_from_notes(notes_text):
    # Vectorize the input text
    X_input = tfidf.transform([notes_text])

    predictions = {}
    for field in fields:
        # Check model type
        clf_path = f"models/{field}_classifier.pkl"
        reg_path = f"models/{field}_regressor.pkl"

        try:
            if os.path.exists(clf_path):
                model = joblib.load(clf_path)
                label_encoder = joblib.load(f"models/{field}_label_encoder.pkl")
                pred = model.predict(X_input)
                predictions[field] = label_encoder.inverse_transform(pred)[0]

            elif os.path.exists(reg_path):
                model = joblib.load(reg_path)
                pred = model.predict(X_input)
                predictions[field] = round(float(pred[0]), 3)

        except Exception as e:
            predictions[field] = f"Error: {str(e)}"

    return predictions

# === Test the prediction function ===
if __name__ == "__main__":
    sample_text = input("Enter earthquake field Notes text:\n> ")
    output = predict_from_notes(sample_text)
    print("\n=== Predicted Structured Fields ===")
    for key, value in output.items():
        print(f"{key}: {value}")
