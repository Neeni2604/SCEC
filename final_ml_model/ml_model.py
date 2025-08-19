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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
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
    
    def add_plotting_to_main_class():
        """
        Code to add to your main() function in the ConsolidatedEarthquakeML script
        """
        plotting_code = '''
        # Add this to the end of your main() function:
        
        # Create visualizer
        visualizer = EarthquakeMLVisualizer()
        
        # Plot 1: Model accuracies
        visualizer.plot_model_accuracies(results, 'earthquake_ml_accuracies.png')
        
        # Plot 2: Prediction scatter for best performing field
        best_field = max(results.keys(), key=lambda k: results[k])
        print(f"Creating detailed analysis for best performing field: {best_field}")
        visualizer.plot_prediction_scatter(ml_model, X, y, best_field, 
                                        f'prediction_analysis_{best_field}.png')
        
        # Plot 3: Comprehensive summary
        visualizer.plot_model_summary(ml_model, X, y, results, 'earthquake_ml_summary.png')
        
        print("\\nAll visualizations have been generated and saved!")
        '''
        return plotting_code
    

class EarthquakeMLVisualizer:
    def __init__(self):
        # Set up colorblind-friendly palette
        # Using Cividis and Set2 palettes which are colorblind accessible
        self.colors = {
            'primary': '#1f77b4',      # Blue
            'secondary': '#ff7f0e',    # Orange  
            'success': '#2ca02c',      # Green
            'warning': '#d62728',      # Red
            'info': '#9467bd',         # Purple
            'accent': '#8c564b',       # Brown
            'light': '#e377c2',        # Pink
            'dark': '#7f7f7f'          # Gray
        }
        
        # Colorblind-friendly palette for multiple categories
        self.palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette(self.palette)
    
    def plot_model_accuracies(self, results, save_path='model_accuracies.png'):
        """
        Create a horizontal bar chart showing accuracy for each field
        """
        # Prepare data
        fields = list(results.keys())
        accuracies = list(results.values())
        
        # Clean field names for display
        display_names = [field.replace('_', ' ').title() for field in fields]
        
        # Sort by accuracy (descending)
        sorted_data = sorted(zip(display_names, accuracies), key=lambda x: x[1], reverse=True)
        sorted_names, sorted_accuracies = zip(*sorted_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create color gradient based on accuracy
        colors = []
        for acc in sorted_accuracies:
            if acc >= 0.8:
                colors.append(self.colors['success'])
            elif acc >= 0.6:
                colors.append(self.colors['primary'])
            elif acc >= 0.4:
                colors.append(self.colors['secondary'])
            else:
                colors.append(self.colors['warning'])
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(sorted_names)), sorted_accuracies, 
                      color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        
        # Customize the plot
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names, fontsize=10)
        ax.set_xlabel('Accuracy Score', fontsize=12, fontweight='bold')
        ax.set_title('Machine Learning Model Accuracy by Geological Field\n' + 
                    'Predicting Characteristics from Field Descriptions', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add accuracy values on bars
        for i, (bar, acc) in enumerate(zip(bars, sorted_accuracies)):
            ax.text(acc + 0.01, i, f'{acc:.3f}', 
                   va='center', ha='left', fontweight='bold', fontsize=9)
        
        # Add vertical lines for reference
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=0.9, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add legend for color coding
        legend_elements = [
            mpatches.Patch(color=self.colors['success'], label='Excellent (≥80%)'),
            mpatches.Patch(color=self.colors['primary'], label='Good (60-80%)'),
            mpatches.Patch(color=self.colors['secondary'], label='Fair (40-60%)'),
            mpatches.Patch(color=self.colors['warning'], label='Poor (<40%)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Set x-axis limits
        ax.set_xlim(0, 1.1)
        
        # Add grid
        ax.grid(True, axis='x', alpha=0.3)
        
        # Improve layout
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Accuracy plot saved as: {save_path}")
        
        plt.show()
        
        return fig, ax
    
    def plot_prediction_scatter(self, ml_model, X, y, field_name, save_path=None):
        """
        Create a scatter plot comparing actual vs predicted values for a specific field
        """
        if field_name not in ml_model.models:
            print(f"No trained model found for field: {field_name}")
            return None, None
        
        # Get the model and data
        model = ml_model.models[field_name]
        label_encoder = ml_model.label_encoders[field_name]
        
        # Prepare data
        X_vectorized = ml_model.vectorizer.transform(X)
        y_actual = y[field_name]
        y_encoded_actual = label_encoder.transform(y_actual)
        
        # Make predictions on the full dataset
        y_pred_encoded = model.predict(X_vectorized)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        
        # Get unique categories and assign colors
        unique_categories = label_encoder.classes_
        n_categories = len(unique_categories)
        
        # Use a subset of our colorblind-friendly palette
        category_colors = self.palette[:min(n_categories, len(self.palette))]
        if n_categories > len(self.palette):
            # Generate additional colors if needed
            additional_colors = plt.cm.Set3(np.linspace(0, 1, n_categories - len(self.palette)))
            category_colors.extend(additional_colors)
        
        # Create color mapping
        color_map = dict(zip(unique_categories, category_colors))
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: Confusion Matrix as Scatter Plot
        # Create numeric mapping for scatter plot
        cat_to_num = {cat: i for i, cat in enumerate(unique_categories)}
        
        x_coords = [cat_to_num[cat] for cat in y_actual]
        y_coords = [cat_to_num[cat] for cat in y_pred]
        
        # Add jitter to see overlapping points
        jitter_strength = 0.1
        x_jittered = [x + np.random.uniform(-jitter_strength, jitter_strength) for x in x_coords]
        y_jittered = [y + np.random.uniform(-jitter_strength, jitter_strength) for y in y_coords]
        
        # Color points by whether prediction is correct
        point_colors = [self.colors['success'] if actual == pred else self.colors['warning'] 
                       for actual, pred in zip(y_actual, y_pred)]
        
        # Create scatter plot
        scatter = ax1.scatter(x_jittered, y_jittered, c=point_colors, alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
        
        # Add diagonal line for perfect predictions
        ax1.plot([0, n_categories-1], [0, n_categories-1], 'k--', alpha=0.5, linewidth=2, label='Perfect Prediction')
        
        # Customize first plot
        ax1.set_xticks(range(n_categories))
        ax1.set_yticks(range(n_categories))
        ax1.set_xticklabels(unique_categories, rotation=45, ha='right')
        ax1.set_yticklabels(unique_categories)
        ax1.set_xlabel('Actual Categories', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Predicted Categories', fontsize=12, fontweight='bold')
        ax1.set_title(f'Actual vs Predicted: {field_name.replace("_", " ").title()}', 
                     fontsize=14, fontweight='bold')
        
        # Add legend for correct/incorrect predictions
        correct_patch = mpatches.Patch(color=self.colors['success'], label='Correct Prediction')
        incorrect_patch = mpatches.Patch(color=self.colors['warning'], label='Incorrect Prediction')
        ax1.legend(handles=[correct_patch, incorrect_patch], loc='upper left')
        
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Category Distribution Comparison
        actual_counts = pd.Series(y_actual).value_counts()
        pred_counts = pd.Series(y_pred).value_counts()
        
        # Ensure all categories are represented
        all_categories = unique_categories
        actual_counts = actual_counts.reindex(all_categories, fill_value=0)
        pred_counts = pred_counts.reindex(all_categories, fill_value=0)
        
        x = np.arange(len(all_categories))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, actual_counts.values, width, 
                       label='Actual', color=self.colors['primary'], alpha=0.8)
        bars2 = ax2.bar(x + width/2, pred_counts.values, width,
                       label='Predicted', color=self.colors['secondary'], alpha=0.8)
        
        # Customize second plot
        ax2.set_xlabel('Categories', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax2.set_title(f'Distribution Comparison: {field_name.replace("_", " ").title()}', 
                     fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(all_categories, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.annotate(f'{int(height)}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)
        
        # Overall title
        fig.suptitle(f'Prediction Analysis for {field_name.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            save_path = f'prediction_scatter_{field_name}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Prediction scatter plot saved as: {save_path}")
        
        plt.show()
        
        return fig, (ax1, ax2)
    
    def plot_model_summary(self, ml_model, X, y, results, save_path='ml_summary.png'):
        """
        Create a comprehensive summary plot with multiple panels
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[2, 1, 1])
        
        # Panel 1: Accuracy bars (spans two columns)
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Prepare accuracy data
        fields = list(results.keys())
        accuracies = list(results.values())
        display_names = [field.replace('_', ' ').title() for field in fields]
        
        # Sort by accuracy
        sorted_data = sorted(zip(display_names, accuracies), key=lambda x: x[1], reverse=True)
        sorted_names, sorted_accuracies = zip(*sorted_data)
        
        # Color based on performance
        colors = []
        for acc in sorted_accuracies:
            if acc >= 0.8:
                colors.append(self.colors['success'])
            elif acc >= 0.6:
                colors.append(self.colors['primary'])
            elif acc >= 0.4:
                colors.append(self.colors['secondary'])
            else:
                colors.append(self.colors['warning'])
        
        bars = ax1.barh(range(len(sorted_names)), sorted_accuracies, color=colors, alpha=0.8)
        ax1.set_yticks(range(len(sorted_names)))
        ax1.set_yticklabels(sorted_names)
        ax1.set_xlabel('Accuracy Score')
        ax1.set_title('Model Performance by Field', fontweight='bold', fontsize=14)
        
        # Add accuracy values
        for i, (bar, acc) in enumerate(zip(bars, sorted_accuracies)):
            ax1.text(acc + 0.01, i, f'{acc:.3f}', va='center', fontweight='bold')
        
        # Panel 2: Data coverage
        ax2 = fig.add_subplot(gs[0, 2])
        
        coverage_data = []
        coverage_labels = []
        for field in fields:
            if field in y:
                non_unknown = sum(y[field] != 'Unknown')
                total = len(y[field])
                coverage = non_unknown / total
                coverage_data.append(coverage)
                coverage_labels.append(field.replace('_', ' ').title()[:15] + '...' if len(field) > 15 else field.replace('_', ' ').title())
        
        # Pie chart for average data coverage
        avg_coverage = np.mean(coverage_data)
        coverage_pie = [avg_coverage, 1 - avg_coverage]
        colors_pie = [self.colors['primary'], self.colors['light']]
        
        ax2.pie(coverage_pie, labels=['Available Data', 'Missing/Unknown'], 
               colors=colors_pie, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Average Data Coverage', fontweight='bold')
        
        # Panel 3: Model complexity (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        
        complexity_data = []
        complexity_labels = []
        for field in fields:
            if field in y:
                unique_classes = len(y[field].unique())
                complexity_data.append(unique_classes)
                complexity_labels.append(field.replace('_', ' ').title())
        
        bars3 = ax3.bar(range(len(complexity_data)), complexity_data, 
                       color=self.palette[:len(complexity_data)], alpha=0.8)
        ax3.set_xticks(range(len(complexity_labels)))
        ax3.set_xticklabels(complexity_labels, rotation=45, ha='right')
        ax3.set_ylabel('Number of Categories')
        ax3.set_title('Model Complexity (Categories per Field)', fontweight='bold')
        
        # Panel 4: Performance distribution (bottom middle)
        ax4 = fig.add_subplot(gs[1, 1])
        
        ax4.hist(accuracies, bins=10, color=self.colors['primary'], alpha=0.7, edgecolor='white')
        ax4.axvline(np.mean(accuracies), color=self.colors['warning'], 
                   linestyle='--', linewidth=2, label=f'Mean: {np.mean(accuracies):.3f}')
        ax4.set_xlabel('Accuracy Score')
        ax4.set_ylabel('Number of Models')
        ax4.set_title('Accuracy Distribution', fontweight='bold')
        ax4.legend()
        
        # Panel 5: Sample text analysis (bottom right)
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Get text length statistics
        text_lengths = [len(desc) for desc in X]
        ax5.boxplot(text_lengths, patch_artist=True, 
                   boxprops=dict(facecolor=self.colors['accent'], alpha=0.7))
        ax5.set_ylabel('Description Length (characters)')
        ax5.set_title('Field Description Lengths', fontweight='bold')
        ax5.set_xticklabels(['All Descriptions'])
        
        # Overall title
        fig.suptitle('Earthquake Field Data ML Model - Comprehensive Analysis', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Summary plot saved as: {save_path}")
        
        plt.show()
        
        return fig


def main():
    print("Consolidated Earthquake Field Data ML Model")
    print("=" * 50)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize model
    ml_model = ConsolidatedEarthquakeML()
    
    # Load and prepare data
    csv_file = 'consolidated_earthquake_observations_20250808.csv'  # Update path as needed
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
    
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    # CREATE PLOTS - UPDATED SECTION TO GENERATE ALL SCATTER PLOTS
    try:
        # Initialize visualizer
        visualizer = EarthquakeMLVisualizer()
        
        # Plot 1: Model accuracies bar chart
        print("Creating accuracy comparison chart...")
        visualizer.plot_model_accuracies(results, 'earthquake_ml_accuracies.png')
        
        # Plot 2: Generate scatter plots for ALL trained models
        print(f"\nGenerating scatter plots for all {len(ml_model.models)} trained models...")
        generated_plots = []
        
        for i, field in enumerate(ml_model.models.keys(), 1):
            print(f"  [{i}/{len(ml_model.models)}] Creating scatter plot for: {field}")
            try:
                filename = f'prediction_scatter_{field}.png'
                visualizer.plot_prediction_scatter(ml_model, X, y, field, filename)
                generated_plots.append(filename)
            except Exception as e:
                print(f"    Error creating scatter plot for {field}: {e}")
        
        # Plot 3: Comprehensive summary dashboard
        print("\nCreating comprehensive summary dashboard...")
        visualizer.plot_model_summary(ml_model, X, y, results, 'earthquake_ml_summary.png')
        
        print(f"\nAll visualizations generated successfully!")
        print(f"Generated files:")
        print(f"  - earthquake_ml_accuracies.png (accuracy bar chart)")
        print(f"  - earthquake_ml_summary.png (comprehensive dashboard)")
        print(f"  - {len(generated_plots)} scatter plot files:")
        for plot_file in generated_plots:
            print(f"    - {plot_file}")
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        print("Continuing without visualizations...")
    
    # Example predictions on sample descriptions
    print(f"\n{'='*60}")
    print("SAMPLE PREDICTIONS")
    print(f"{'='*60}")
    
    # Test with a few sample descriptions if available
    sample_descriptions = [
        "Observed surface rupture with right-lateral offset of approximately 15 cm. Clear scarp facing north with visible gouge material in fault zone.",
        "Ground cracks observed with minor vertical displacement. No clear fault orientation visible. Sandy soil conditions.",
        "Well-defined fault scarp with striations visible on fault plane. Left-lateral motion evident from offset features."
    ]
    
    for i, desc in enumerate(sample_descriptions, 1):
        print(f"\nSample {i}: {desc[:80]}...")
        predictions = ml_model.predict_from_description(desc)
        
        print("Predictions:")
        for field, pred_info in predictions.items():
            if 'error' not in pred_info:
                print(f"  {field.replace('_', ' ').title()}: {pred_info['prediction']} (confidence: {pred_info['confidence']:.2f})")
            else:
                print(f"  {field.replace('_', ' ').title()}: Error - {pred_info['error']}")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples processed: {len(X)}")
    print(f"Models trained: {successful_models}")
    print(f"Average model performance: {np.mean(list(results.values())):.3f}" if results else "N/A")
    print(f"Scatter plots generated: {len(generated_plots) if 'generated_plots' in locals() else 0}")
    print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()