import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Load the consolidated dataset, which contains both Napa and Ridgecrest observations 
df = pd.read_csv("consolidated_earthquake_observations_20250703.csv")

# Filter rows that have both "Notes" and "Slip_Sense"
# We only waant rows that have a non-empty "Notes" field and a valid "Slip_Sense" label 
df = df[df["Notes"].notnull() & df["Slip_Sense"].notnull()]
df = df[df["Notes"].str.strip() != ""] # Remove empty notes
df = df[df["Slip_Sense"].str.strip() != ""] # Remove empty slip sense labels

# Define features (X) and target variable (y)
# X will be the free-text "Notes" field (which is our input)
# y will be the structuerd "Slip_Sense" field (our output label)
X_text = df["Notes"]
y_labels = df["Slip_Sense"]

# Convert the text into TF_IDF features 
# This converts the Notes into a numerical representation using: 
# - max 100 features (words/phrases)
# - unigrams and bigrams (1- and 2-word combinations)
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(X_text)

# Encode the "Slip_Sense" labels numerically
# Converts categories like "Left-lateral" -> 0, 'Right-lateral' -> 1, etc. 
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_labels)

# Split the data into training and test sets 
# Train on 80% of the data, test on the remaininig 20%
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y_encoded, test_size=0.2, random_state=42
)

# Train a Random Forest Classifier
# Random Forest is a good first model because it handles non-linearities and is robust to overfitting
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

#Predict and evaluate the model
y_pred = clf.predict(X_test)

#Print overall accuract and per-class precision/recall
print ("Accuracy:", accuracy_score(y_test,y_pred))
print("\nClassification Report:\n") 
print(classification_report(y_test, y_pred, target_names=label_encoder.classes))
# Save the model and vectorizer for future use
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.multiclass import unique_labels

print("LOADED: baseline_slip_sense_model.py ✅")

# Load the consolidated dataset, which contains both Napa and Ridgecrest observations 
df = pd.read_csv("consolidated_earthquake_observations_20250703.csv")

# Filter rows that have both "Notes" and "Slip_Sense"
# We only waant rows that have a non-empty "Notes" field and a valid "Slip_Sense" label 
df = df[df["Notes"].notnull() & df["Slip_Sense"].notnull()]
df = df[df["Notes"].str.strip() != ""] # Remove empty notes
df = df[df["Slip_Sense"].str.strip() != ""] # Remove empty slip sense labels

# Define features (X) and target variable (y)
# X will be the free-text "Notes" field (which is our input)
# y will be the structuerd "Slip_Sense" field (our output label)
X_text = df["Notes"]
y_labels = df["Slip_Sense"]

# Convert the text into TF_IDF features 
# This converts the Notes into a numerical representation using: 
# - max 100 features (words/phrases)
# - unigrams and bigrams (1- and 2-word combinations)
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(X_text)

# Encode the "Slip_Sense" labels numerically
# Converts categories like "Left-lateral" -> 0, 'Right-lateral' -> 1, etc. 
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_labels)

# Split the data into training and test sets 
# Train on 80% of the data, test on the remaininig 20%
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y_encoded, test_size=0.2, random_state=42
)

# Train a Random Forest Classifier
# Random Forest is a good first model because it handles non-linearities and is robust to overfitting
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

from sklearn.metrics import classification_report, accuracy_score

# Predict and evaluate the model
y_pred = clf.predict(X_test)

# Print overall accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Debug: Show available class labels
print("Label classes:", list(label_encoder.classes_))

# Print classification report
print("\nClassification Report:\n")
labels = unique_labels(y_test, y_pred)
print(classification_report(
    y_test, 
    y_pred, 
    labels=labels,
    target_names=label_encoder.classes_[labels],
    zero_division=0
))
# Save the trained model and vectorizer for future use
