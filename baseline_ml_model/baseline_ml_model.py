"""
Enhanced ML Model for Earthquake Field Data - Current Schema
Predicts multiple geological and metadata fields from Notes field
Uses combined Napa (2014) and Ridgecrest (2019) earthquake datasets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import re
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class EnhancedEarthquakeFieldDataML:
    """Enhanced ML model for predicting earthquake field data from Notes"""
    
    def __init__(self):
        # Enhanced text vectorizer with optimized parameters
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams for better context
            lowercase=True,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        self.models = {}
        self.label_encoders = {}
        self.feature_importances = {}
        
        # Target fields organized by category for better analysis
        self.target_fields = {
            'geological': [
                'Feature_Origin',
                'Slip_Sense', 
                'Scarp_Facing_Direction',
                'Rupture_Expression',
                '_feature_type',
                '_gouge_observed',
                '_striations_observed'
            ],
            'measurements': [
                'Local_Fault_Azimuth_Degrees',
                'Local_Fault_Dip',
                'Heave_cm',
                'Horizontal_Separation_cm',
                'Vertical_Separation_cm',
                'Net_Slip_Preferred_cm'
            ],
            'metadata': [
                'Creator',
                '_obs_affiliation',
                '_team',
                '_observed_feature',
                '_source'
            ]
        }
        
        # Flatten all target fields
        self.all_target_fields = []
        for category in self.target_fields.values():
            self.all_target_fields.extend(category)

    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Remove source prefix that was added during mapping
        text = re.sub(r'^Source:\s*\w+\s*\d{4};\s*', '', text)
        
        # Clean common field notation patterns
        text = re.sub(r'\b\d+[°]\b', ' degrees ', text)  # Replace degree symbols
        text = re.sub(r'\b\d+\s*cm\b', ' centimeters ', text)  # Standardize cm references
        text = re.sub(r'\b\d+\s*m\b', ' meters ', text)  # Standardize m references
        
        # Standardize directional terms
        directional_mapping = {
            r'\bNE\b': 'northeast',
            r'\bNW\b': 'northwest', 
            r'\bSE\b': 'southeast',
            r'\bSW\b': 'southwest',
            r'\bN\b': 'north',
            r'\bS\b': 'south',
            r'\bE\b': 'east',
            r'\bW\b': 'west'
        }
        for pattern, replacement in directional_mapping.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Clean and normalize
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def categorize_numeric_field(self, series, field_name):
        """Convert numeric fields to categorical ranges for better prediction"""
        def categorize_value(val):
            if pd.isna(val) or val == '' or str(val).lower() in ['nan', 'unknown', 'null']:
                return 'UNKNOWN'
            
            try:
                # Extract numeric value
                if isinstance(val, str):
                    # Handle ranges like "1-2", ">5", "<1", "~3"
                    val_clean = re.sub(r'[^\d.]', '', val)
                    if not val_clean:
                        return 'UNKNOWN'
                    val = float(val_clean)
                
                val = float(val)
                
                # Categorize based on field type
                if 'azimuth' in field_name.lower() or 'degrees' in field_name.lower():
                    # Azimuth categorization
                    if 0 <= val < 45 or 315 <= val <= 360:
                        return 'N (0-45°, 315-360°)'
                    elif 45 <= val < 135:
                        return 'E (45-135°)'
                    elif 135 <= val < 225:
                        return 'S (135-225°)'
                    elif 225 <= val < 315:
                        return 'W (225-315°)'
                    else:
                        return 'UNKNOWN'
                        
                elif 'dip' in field_name.lower():
                    # Dip categorization
                    if val < 30:
                        return 'Shallow (<30°)'
                    elif val < 60:
                        return 'Moderate (30-60°)'
                    elif val <= 90:
                        return 'Steep (60-90°)'
                    else:
                        return 'UNKNOWN'
                        
                else:
                    # General measurement categorization
                    if val == 0:
                        return 'None (0)'
                    elif val < 1:
                        return 'Very Small (<1)'
                    elif val < 5:
                        return 'Small (1-5)'
                    elif val < 15:
                        return 'Medium (5-15)'
                    elif val < 50:
                        return 'Large (15-50)'
                    else:
                        return 'Very Large (>50)'
                        
            except (ValueError, TypeError):
                return 'UNKNOWN'
        
        return series.apply(categorize_value)

    def load_and_prepare_data(self, napa_file, ridgecrest_file):
        """Load and prepare combined dataset"""
        print("Loading earthquake field data from current schema files...")
        
        # Load datasets
        try:
            napa_df = pd.read_csv(napa_file, encoding='utf-8')
            ridgecrest_df = pd.read_csv(ridgecrest_file, encoding='utf-8')
        except UnicodeDecodeError:
            napa_df = pd.read_csv(napa_file, encoding='latin-1')
            ridgecrest_df = pd.read_csv(ridgecrest_file, encoding='latin-1')
        
        print(f"Loaded Napa: {len(napa_df)} records")
        print(f"Loaded Ridgecrest: {len(ridgecrest_df)} records")
        
        # Add dataset source identifier
        napa_df['dataset_source'] = 'Napa_2014'
        ridgecrest_df['dataset_source'] = 'Ridgecrest_2019'
        
        # Combine datasets
        combined_df = pd.concat([napa_df, ridgecrest_df], ignore_index=True, sort=False)
        print(f"Combined dataset: {len(combined_df)} records")
        
        # Filter records with substantial Notes content
        combined_df = combined_df[
            combined_df['Notes'].notna() & 
            (combined_df['Notes'].str.len() > 20)
        ].copy()
        
        print(f"Records with substantial Notes: {len(combined_df)}")
        
        # Prepare features (Notes field)
        X = combined_df['Notes'].apply(self.preprocess_text)
        
        # Prepare targets
        y = {}
        for field in self.all_target_fields:
            if field in combined_df.columns:
                if field in ['Local_Fault_Azimuth_Degrees', 'Local_Fault_Dip', 'Heave_cm', 
                            'Horizontal_Separation_cm', 'Vertical_Separation_cm', 'Net_Slip_Preferred_cm']:
                    # Categorize numeric fields
                    y[field] = self.categorize_numeric_field(combined_df[field], field)
                else:
                    # Keep categorical fields as-is, but clean them
                    y[field] = combined_df[field].fillna('UNKNOWN').astype(str)
                    y[field] = y[field].replace(['nan', 'None', ''], 'UNKNOWN')
            else:
                print(f"Warning: Field '{field}' not found in data")
        
        # Add dataset source as a target to analyze cross-dataset performance
        y['dataset_source'] = combined_df['dataset_source']
        
        return X, y, combined_df

    def train_models(self, X, y):
        """Train ML models for each target field"""
        print("\nTraining enhanced ML models...")
        
        # Vectorize text
        print("Vectorizing text with enhanced TF-IDF...")
        X_vectorized = self.vectorizer.fit_transform(X)
        print(f"Text features shape: {X_vectorized.shape}")
        
        trained_count = 0
        for field in self.all_target_fields + ['dataset_source']:
            if field not in y:
                continue
                
            print(f"\nTraining model for '{field}'...")
            
            # Encode labels
            self.label_encoders[field] = LabelEncoder()
            y_encoded = self.label_encoders[field].fit_transform(y[field])
            
            # Skip if insufficient classes
            unique_classes = len(np.unique(y_encoded))
            if unique_classes < 2:
                print(f"Skipping '{field}' - only {unique_classes} class")
                continue
            
            # Use different models based on field type and class count
            if unique_classes <= 5:
                # Use Gradient Boosting for fields with few classes
                base_classifier = GradientBoostingClassifier(
                    n_estimators=100, 
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
            else:
                # Use Random Forest for fields with many classes
                base_classifier = RandomForestClassifier(
                    n_estimators=150,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            
            # Wrap in OneVsRest for multi-class
            model = OneVsRestClassifier(base_classifier, n_jobs=-1)
            
            # Train model
            model.fit(X_vectorized, y_encoded)
            self.models[field] = model
            
            # Store feature importance if available
            if hasattr(base_classifier, 'feature_importances_'):
                self.feature_importances[field] = base_classifier.feature_importances_
            
            trained_count += 1
            print(f"Trained - Classes: {unique_classes}")
        
        print(f"\nSuccessfully trained {trained_count} models!")

    def evaluate_models(self, X, y):
        """Comprehensive model evaluation"""
        print("\n" + "="*70)
        print("ENHANCED MODEL EVALUATION - CURRENT SCHEMA")
        print("="*70)
        
        X_vectorized = self.vectorizer.transform(X)
        
        # Split data for evaluation
        X_train, X_test, indices_train, indices_test = train_test_split(
            X_vectorized, range(len(X)), test_size=0.25, random_state=42, stratify=y['dataset_source']
        )
        
        results = {}
        category_results = {}
        
        # Evaluate each field
        for field in self.models.keys():
            if field not in y:
                continue
                
            y_field = y[field]
            y_encoded = self.label_encoders[field].transform(y_field)
            
            y_train = y_encoded[indices_train]
            y_test = y_encoded[indices_test]
            
            # Train and predict
            model = self.models[field]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            results[field] = accuracy
            
            # Cross-validation score
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                cv_mean = cv_scores.mean()
            except:
                cv_mean = accuracy
            
            print(f"\n--- {field.upper()} ---")
            print(f"Accuracy: {accuracy:.3f} | CV Score: {cv_mean:.3f}")
            
            # Show detailed report for important fields
            unique_labels = np.unique(y_test)
            if len(unique_labels) <= 15 and field in ['Feature_Origin', 'Slip_Sense', 'Scarp_Facing_Direction']:
                target_names = [self.label_encoders[field].classes_[i] for i in unique_labels]
                report = classification_report(y_test, y_pred, labels=unique_labels, 
                                             target_names=target_names, zero_division=0)
                print("Classification Report:")
                print(report)
        
        # Organize results by category
        for category, fields in self.target_fields.items():
            category_accuracies = [results[field] for field in fields if field in results]
            if category_accuracies:
                category_results[category] = np.mean(category_accuracies)
        
        # Print summary
        print("\n" + "="*70)
        print("PERFORMANCE SUMMARY BY CATEGORY")
        print("="*70)
        
        for category, avg_accuracy in category_results.items():
            print(f"{category.upper()}: {avg_accuracy:.3f}")
            for field in self.target_fields[category]:
                if field in results:
                    print(f"  • {field}: {results[field]:.3f}")
        
        # Overall statistics
        all_accuracies = list(results.values())
        if all_accuracies:
            overall_avg = np.mean(all_accuracies)
            print(f"\n🎯 OVERALL AVERAGE ACCURACY: {overall_avg:.3f}")
            print(f"📊 Best Field: {max(results.items(), key=lambda x: x[1])}")
            print(f"📉 Worst Field: {min(results.items(), key=lambda x: x[1])}")
            
            high_performance = [field for field, acc in results.items() if acc >= 0.8]
            print(f"🔥 High Performance Fields (≥0.8): {len(high_performance)}/{len(results)}")
        
        # Dataset source analysis
        if 'dataset_source' in results:
            dataset_acc = results['dataset_source']
            print(f"\n🔍 Dataset Distinguishability: {dataset_acc:.3f}")
            if dataset_acc > 0.85:
                print("   → Strong linguistic differences between Napa and Ridgecrest datasets")
            elif dataset_acc > 0.7:
                print("   → Moderate linguistic differences between datasets")
            else:
                print("   → Similar linguistic patterns across datasets")
        
        return results

    def predict_from_notes(self, notes_text):
        """Make predictions for new notes text"""
        if not self.models:
            print("No trained models available!")
            return {}
        
        # Preprocess and vectorize
        clean_notes = self.preprocess_text(notes_text)
        notes_vectorized = self.vectorizer.transform([clean_notes])
        
        predictions = {}
        
        for field, model in self.models.items():
            if field == 'dataset_source':  # Skip dataset source in predictions
                continue
                
            # Make prediction
            pred_encoded = model.predict(notes_vectorized)[0]
            pred_class = self.label_encoders[field].inverse_transform([pred_encoded])[0]
            
            # Get confidence
            try:
                pred_proba = model.predict_proba(notes_vectorized)[0]
                confidence = np.max(pred_proba)
            except:
                confidence = 0.5  # Default if probability not available
            
            predictions[field] = {
                'prediction': pred_class,
                'confidence': confidence
            }
        
        return predictions

    def demonstrate_predictions(self, X, y, n_examples=3):
        """Show prediction examples"""
        print("\n" + "="*70)
        print("PREDICTION EXAMPLES")
        print("="*70)
        
        # Select diverse examples
        sample_indices = np.random.choice(len(X), min(n_examples, len(X)), replace=False)
        
        for i, idx in enumerate(sample_indices):
            notes_text = X.iloc[idx]
            
            print(f"\n--- EXAMPLE {i+1} ---")
            print(f"Notes: \"{notes_text[:200]}{'...' if len(notes_text) > 200 else ''}\"")
            
            predictions = self.predict_from_notes(notes_text)
            
            print("\nKey Predictions:")
            important_fields = ['Feature_Origin', 'Slip_Sense', 'Scarp_Facing_Direction', 'Creator']
            
            for field in important_fields:
                if field in predictions and field in y:
                    actual = y[field].iloc[idx]
                    pred_info = predictions[field]
                    predicted = pred_info['prediction']
                    confidence = pred_info['confidence']
                    
                    match = "✓" if predicted == actual else "✗"
                    print(f"  {field}: {predicted} (conf: {confidence:.2f}) {match} Actual: {actual}")

def main():
    """Main execution function"""
    print("Enhanced Earthquake Field Data ML Model")
    print("=" * 70)
    print("Predicting geological and metadata fields from Notes field")
    print("Using combined Napa (2014) and Ridgecrest (2019) current schema data\n")
    
    # Initialize enhanced model
    model = EnhancedEarthquakeFieldDataML()
    
    # Load and prepare data
    X, y, df = model.load_and_prepare_data(
        'napa_current_schema_20250807.csv',
        'ridgecrest_current_schema_20250807.csv'
    )
    
    # Train models
    model.train_models(X, y)
    
    # Evaluate models
    results = model.evaluate_models(X, y)
    
    # Show prediction examples
    model.demonstrate_predictions(X, y, n_examples=3)
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📈 Trained on {len(X)} field observations")
    print(f"🔬 Ready to predict geological features from Notes field")

if __name__ == "__main__":
    main()