"""
Consolidated Earthquake Field Data ML Model
Trains ML models on the consolidated Napa + Ridgecrest dataset
Uses Notes field to predict geological and observational characteristics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
import re
from datetime import datetime
warnings.filterwarnings('ignore')

class ConsolidatedEarthquakeML:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=2000,  # Increased for larger dataset
            stop_words='english',
            ngram_range=(1, 3),  # Include bigrams and trigrams
            min_df=2,  # Ignore rare terms
            max_df=0.95  # Ignore very common terms
        )
        
        self.models = {}
        self.label_encoders = {}
        
        # Define target fields to predict - focusing on geological/observational characteristics
        self.target_fields = [
            # Core geological characteristics
            'Feature_Origin',                    # Origin of the feature (Tectonic, Uncertain, etc.)
            'Slip_Sense',                       # Sense of slip movement
            'Rupture_Expression',               # How the rupture is expressed
            'Scarp_Facing_Direction',           # Direction the scarp faces
            
            # Observed feature characteristics
            '_observed_feature',                # What feature was observed
            '_feature_type',                    # Natural vs Cultural
            
            # Geological observations
            'gouge_observed_L',                 # Whether gouge was observed
            'striations_observed_L',            # Whether striations were observed
            '_heave_type',                      # Type of heave movement
            '_vert_slip_type',                  # Type of vertical slip
            
            # Measurement types and orientations (categorical versions)
            'Fault_Slip_Measurement_Type',     # Type of slip measurement
            
            # Derived categorical fields for numeric measurements
            'fault_azimuth_category',           # Categorized fault azimuth
            'slip_magnitude_category',          # Categorized slip magnitude
            'offset_magnitude_category',        # Categorized offset magnitude
        ]
        
        # Fields to exclude from prediction (metadata, identifiers, photos, etc.)
        self.excluded_fields = [
            'OBJECTID', 'Station_ID', 'GlobalID',
            'CreationDate', 'EditDate', 'Creator', 'Editor',
            '_napa_photo_exists',  # Photo information
            '_latitude', '_longitude', '_orig_lat', '_orig_lon',  # Coordinates
            '_obs_affiliation', '_team', '_team_id', '_obs_position',  # Observer metadata
            '_source', '_citation',  # Source information
            'Notes',  # This is our input feature
        ]

    def categorize_azimuth(self, azimuth_series):
        """Convert numeric azimuth values to directional categories"""
        def azimuth_to_category(val):
            if pd.isna(val) or val == '' or str(val).lower() in ['nan', 'unknown']:
                return 'Unknown'
            try:
                # Extract numeric value from string if needed
                val_str = str(val).lower()
                numbers = re.findall(r'[\d.]+', val_str)
                if not numbers:
                    return 'Unknown'
                
                val_num = float(numbers[0])
                
                # Convert to compass directions
                if (val_num >= 337.5) or (val_num < 22.5):
                    return 'N'
                elif 22.5 <= val_num < 67.5:
                    return 'NE' 
                elif 67.5 <= val_num < 112.5:
                    return 'E'
                elif 112.5 <= val_num < 157.5:
                    return 'SE'
                elif 157.5 <= val_num < 202.5:
                    return 'S'
                elif 202.5 <= val_num < 247.5:
                    return 'SW'
                elif 247.5 <= val_num < 292.5:
                    return 'W'
                elif 292.5 <= val_num < 337.5:
                    return 'NW'
                else:
                    return 'Unknown'
            except (ValueError, TypeError):
                return 'Unknown'
        
        return azimuth_series.apply(azimuth_to_category)

    def categorize_slip_magnitude(self, slip_series):
        """Convert slip measurements to magnitude categories"""
        def slip_to_category(val):
            if pd.isna(val) or val == '' or str(val).lower() in ['nan', 'unknown']:
                return 'Unknown'
            try:
                val_str = str(val).lower()
                # Extract numeric value
                numbers = re.findall(r'[\d.]+', val_str)
                if not numbers:
                    return 'Unknown'
                
                val_num = float(numbers[0])
                
                if val_num == 0:
                    return 'None'
                elif val_num < 1:
                    return 'Very Small (<1cm)'
                elif val_num < 5:
                    return 'Small (1-5cm)'
                elif val_num < 20:
                    return 'Medium (5-20cm)'
                elif val_num < 50:
                    return 'Large (20-50cm)'
                else:
                    return 'Very Large (>50cm)'
            except (ValueError, TypeError):
                return 'Unknown'
        
        return slip_series.apply(slip_to_category)

    def categorize_offset_magnitude(self, offset_series):
        """Convert offset measurements to magnitude categories"""
        def offset_to_category(val):
            if pd.isna(val) or val == '' or str(val).lower() in ['nan', 'unknown']:
                return 'Unknown'
            try:
                val_str = str(val).lower()
                numbers = re.findall(r'[\d.]+', val_str)
                if not numbers:
                    return 'Unknown'
                
                val_num = float(numbers[0])
                
                if val_num == 0:
                    return 'None'
                elif val_num < 2:
                    return 'Very Small (<2cm)'
                elif val_num < 10:
                    return 'Small (2-10cm)'
                elif val_num < 30:
                    return 'Medium (10-30cm)'
                elif val_num < 100:
                    return 'Large (30-100cm)'
                else:
                    return 'Very Large (>100cm)'
            except (ValueError, TypeError):
                return 'Unknown'
        
        return offset_series.apply(offset_to_category)

    def load_and_prepare_data(self, csv_file):
        """Load and prepare the consolidated earthquake field data"""
        print("Loading consolidated earthquake field data...")
        
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_file, encoding='latin-1')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_file, encoding='cp1252', errors='ignore')
        
        print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Filter for rows with meaningful notes (excluding just source info)
        df = df[df['Notes'].notna() & (df['Notes'].str.len() > 20)]
        print(f"After filtering for meaningful notes: {len(df)} rows")
        
        # Create derived categorical features
        print("Creating derived categorical features...")
        
        # Fault azimuth category from Local_Fault_Azimuth_Degrees
        df['fault_azimuth_category'] = self.categorize_azimuth(df['Local_Fault_Azimuth_Degrees'])
        
        # Slip magnitude categories from various slip fields
        slip_fields = ['Net_Slip_Preferred_cm', 'Net_Slip_Preffered_cm', 'Horizontal_Separation_cm']
        for field in slip_fields:
            if field in df.columns:
                df['slip_magnitude_category'] = self.categorize_slip_magnitude(df[field])
                break
        
        # Offset magnitude category from vertical separation
        if 'Vertical_Separation_cm' in df.columns:
            df['offset_magnitude_category'] = self.categorize_offset_magnitude(df['Vertical_Separation_cm'])
        
        # Clean up text in Notes field - remove source prefixes for better analysis
        def clean_notes(note):
            if pd.isna(note):
                return ''
            # Remove source prefixes but keep the actual descriptive content
            note = str(note)
            if note.startswith('Source: Napa 2014;'):
                note = note.replace('Source: Napa 2014;', '').strip()
            elif note.startswith('Source: Ridgecrest 2019;'):
                note = note.replace('Source: Ridgecrest 2019;', '').strip()
            return note
        
        df['Notes_cleaned'] = df['Notes'].apply(clean_notes)
        
        # Prepare features (cleaned notes)
        X = df['Notes_cleaned'].fillna('')
        
        # Prepare targets
        y = {}
        target_stats = {}
        
        for field in self.target_fields:
            if field in df.columns:
                # Clean and prepare target field
                y[field] = df[field].fillna('Unknown').astype(str)
                
                # Get statistics
                value_counts = y[field].value_counts()
                target_stats[field] = {
                    'total_values': len(y[field]),
                    'unique_values': len(value_counts),
                    'non_unknown': sum(y[field] != 'Unknown'),
                    'top_values': dict(value_counts.head(5))
                }
            else:
                print(f"Warning: Field '{field}' not found in data")
        
        # Print target statistics
        print("\nTarget field statistics:")
        for field, stats in target_stats.items():
            print(f"\n{field}:")
            print(f"  Total values: {stats['total_values']}")
            print(f"  Unique values: {stats['unique_values']}")
            print(f"  Non-unknown values: {stats['non_unknown']}")
            print(f"  Data coverage: {stats['non_unknown']/stats['total_values']:.1%}")
            print(f"  Top values: {stats['top_values']}")
        
        return X, y, df

    def train_models(self, X, y):
        """Train ML models for each target field"""
        print(f"\nTraining models on {len(X)} samples...")
        
        # Vectorize text descriptions with enhanced features
        print("Vectorizing text descriptions...")
        X_vectorized = self.vectorizer.fit_transform(X)
        print(f"Text features shape: {X_vectorized.shape}")
        
        successful_models = 0
        
        # Train a model for each target field
        for field in self.target_fields:
            if field not in y:
                continue
                
            print(f"\nTraining model for '{field}'...")
            
            # Encode labels
            self.label_encoders[field] = LabelEncoder()
            y_encoded = self.label_encoders[field].fit_transform(y[field])
            
            # Get class statistics
            unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
            non_unknown_count = sum(y[field] != 'Unknown')
            min_class_size = min(class_counts)
            
            # Skip if insufficient data for meaningful training
            if len(unique_classes) < 2:
                print(f"Skipped: Only {len(unique_classes)} class found")
                continue
            elif non_unknown_count < 15:  # Increased threshold
                print(f"Skipped: Only {non_unknown_count} non-unknown samples (need ≥15)")
                continue
            elif min_class_size == 1 and len(unique_classes) > 10:
                print(f"Skipped: Too many singleton classes ({len(unique_classes)} classes, min size: {min_class_size})")
                continue
            
            # Choose model based on problem complexity and data availability
            if len(unique_classes) <= 5 and non_unknown_count >= 100:
                # Use Gradient Boosting for simpler problems with good data
                base_classifier = GradientBoostingClassifier(
                    n_estimators=100, 
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                model_type = "Gradient Boosting"
            else:
                # Use Random Forest for complex problems or limited data
                base_classifier = RandomForestClassifier(
                    n_estimators=200, 
                    max_depth=10,
                    min_samples_split=max(5, min_class_size),  # Adjust based on min class size
                    min_samples_leaf=max(2, min_class_size // 2),
                    random_state=42
                )
                model_type = "Random Forest"
            
            # Use appropriate classifier setup
            if len(unique_classes) == 2:
                # Binary classification
                model = base_classifier
            else:
                # Multi-class classification
                model = OneVsRestClassifier(base_classifier)
            
            try:
                model.fit(X_vectorized, y_encoded)
                self.models[field] = model
                successful_models += 1
                
                print(f"Trained successfully using {model_type}")
                print(f"    Classes: {len(unique_classes)}, Informative samples: {non_unknown_count}, Min class size: {min_class_size}")
                
            except Exception as e:
                print(f"Training failed: {e}")
                continue
        
        print(f"\nSuccessfully trained {successful_models} models!")
        return successful_models

    def evaluate_models(self, X, y):
        """Evaluate model performance using cross-validation and hold-out testing"""
        print(f"\n{'='*60}")
        print("MODEL EVALUATION")
        print(f"{'='*60}")
        
        # Vectorize text
        X_vectorized = self.vectorizer.transform(X)
        
        results = {}
        
        for field in self.models.keys():
            print(f"\n--- {field.upper().replace('_', ' ')} ---")
            
            # Get target data
            y_field = y[field]
            y_encoded = self.label_encoders[field].transform(y_field)
            
            model = self.models[field]
            
            # Check class distribution for stratification
            unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
            min_class_size = min(class_counts)
            
            print(f"Classes: {len(unique_classes)}, Min class size: {min_class_size}")
            
            # Cross-validation scores (handle small classes)
            try:
                if min_class_size >= 5:
                    cv_scores = cross_val_score(model, X_vectorized, y_encoded, cv=5, scoring='accuracy')
                elif min_class_size >= 3:
                    cv_scores = cross_val_score(model, X_vectorized, y_encoded, cv=3, scoring='accuracy')
                else:
                    cv_scores = cross_val_score(model, X_vectorized, y_encoded, cv=2, scoring='accuracy')
                
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                print(f"Cross-validation accuracy: {cv_mean:.3f} (±{cv_std:.3f})")
            except Exception as e:
                print(f"Cross-validation failed: {e}")
                cv_mean = 0
            
            # Hold-out test with careful handling of small classes
            try:
                # Use stratification only if all classes have at least 2 samples
                if min_class_size >= 2:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_vectorized, y_encoded, test_size=0.2, random_state=42, 
                        stratify=y_encoded
                    )
                else:
                    # No stratification for classes with only 1 sample
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_vectorized, y_encoded, test_size=0.2, random_state=42
                    )
                
                # Retrain model on training set
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                results[field] = accuracy
                
                print(f"Hold-out test accuracy: {accuracy:.3f}")
                
                # Show class distribution in test set
                unique_labels = np.unique(np.concatenate([y_test, y_pred]))
                target_names = [self.label_encoders[field].classes_[i] for i in unique_labels]
                
                # Only show detailed report for models with reasonable performance and manageable classes
                if accuracy > 0.3 and len(unique_labels) <= 15 and len(y_test) > 10:
                    try:
                        report = classification_report(
                            y_test, y_pred, 
                            labels=unique_labels, 
                            target_names=target_names, 
                            zero_division=0
                        )
                        print(f"\nDetailed Classification Report:")
                        print(report)
                    except Exception as e:
                        print(f"Could not generate detailed report: {e}")
                        
            except Exception as e:
                print(f"Hold-out test failed: {e}")
                # Fallback: use full dataset performance
                y_pred_full = model.predict(X_vectorized)
                accuracy = accuracy_score(y_encoded, y_pred_full)
                results[field] = accuracy
                print(f"Full dataset accuracy (fallback): {accuracy:.3f}")
        
        # Summary
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for field, accuracy in sorted_results:
            print(f"{field.replace('_', ' ').title():<35}: {accuracy:.3f}")
        
        if results:
            avg_accuracy = np.mean(list(results.values()))
            print(f"\nOverall Average Accuracy: {avg_accuracy:.3f}")
        
        return results

    def predict_from_description(self, description):
        """Make predictions for a single description"""
        if not self.models:
            print("No trained models available!")
            return {}
        
        # Clean description
        cleaned_desc = description.replace('Source: Napa 2014;', '').replace('Source: Ridgecrest 2019;', '').strip()
        
        # Vectorize description
        desc_vectorized = self.vectorizer.transform([cleaned_desc])
        
        predictions = {}
        for field, model in self.models.items():
            try:
                # Make prediction
                pred_encoded = model.predict(desc_vectorized)[0]
                
                # Decode prediction
                pred_class = self.label_encoders[field].inverse_transform([pred_encoded])[0]
                
                # Get prediction confidence
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(desc_vectorized)[0]
                    confidence = np.max(pred_proba)
                elif hasattr(model, 'decision_function'):
                    decision_scores = model.decision_function(desc_vectorized)[0]
                    if isinstance(decision_scores, np.ndarray):
                        confidence = np.max(decision_scores) / (np.max(decision_scores) - np.min(decision_scores) + 1e-8)
                    else:
                        confidence = abs(decision_scores) / (abs(decision_scores) + 1)
                else:
                    confidence = 0.5  # No confidence available
                
                predictions[field] = {
                    'prediction': pred_class,
                    'confidence': confidence
                }
            except Exception as e:
                predictions[field] = {
                    'prediction': 'Error',
                    'confidence': 0,
                    'error': str(e)
                }
        
        return predictions

    def demonstrate_predictions(self, X, y, n_samples=5):
        """Demonstrate predictions on sample data"""
        print(f"\n{'='*60}")
        print("PREDICTION EXAMPLES")
        print(f"{'='*60}")
        
        # Sample some descriptions
        sample_indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        
        for i, idx in enumerate(sample_indices):
            description = X.iloc[idx]
            
            print(f"\n--- EXAMPLE {i+1} ---")
            print(f"Description: \"{description[:150]}{'...' if len(description) > 150 else ''}\"")
            
            # Make prediction
            predictions = self.predict_from_description(description)
            
            print(f"\nPredictions:")
            for field, pred_info in predictions.items():
                if 'error' in pred_info:
                    continue
                    
                actual = y[field].iloc[idx] if field in y else 'N/A'
                predicted = pred_info['prediction']
                confidence = pred_info['confidence']
                
                match = "✓" if predicted == actual else "✗"
                print(f"  {field.replace('_', ' ').title():<25}: {predicted:<15} (conf: {confidence:.2f}) {match} Actual: {actual}")

def main():
    print("Consolidated Earthquake Field Data ML Model")
    print("=" * 50)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize model
    ml_model = ConsolidatedEarthquakeML()
    
    # Load and prepare data
    csv_file = 'consolidated_earthquake_observations_20250807.csv'  # Update path as needed
    X, y, df = ml_model.load_and_prepare_data(csv_file)
    
    if len(X) == 0:
        print("No suitable data found for training!")
        return
    
    # Train models
    successful_models = ml_model.train_models(X, y)
    
    if successful_models == 0:
        print("No models could be trained!")
        return
    
    # Evaluate models
    results = ml_model.evaluate_models(X, y)
    
    # Demonstrate predictions
    ml_model.demonstrate_predictions(X, y, n_samples=3)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples processed: {len(X)}")
    print(f"Models trained: {successful_models}")
    print(f"Average model performance: {np.mean(list(results.values())):.3f}" if results else "N/A")
    print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()