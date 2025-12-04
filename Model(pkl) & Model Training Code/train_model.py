import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# 1. Data Loading & Quality Control
# ==========================================
print("Loading dataset...")
try:
    df = pd.read_csv('Crop_recommendation.csv')
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'Crop_recommendation.csv' not found. Please download it from Kaggle.")
    exit()

# Quality Control
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
# Drop duplicates if any
df = df.drop_duplicates()

# ==========================================
# 2. Exploratory Data Analysis (EDA) & Skewness Check
# ==========================================
# Check skewness to decide on preprocessing
numeric_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
skewness = df[numeric_features].skew()
print(f"\nSkewness of features:\n{skewness}")

# Insight: Rainfall and P/K usually have skewness. We will handle this in preprocessing.

# ==========================================
# 3. Feature Engineering (Domain Knowledge)
# ==========================================
# UPL Context: It's not just about the absolute amount of N, but the ratio of N to total nutrients.
# Creating a 'Total Nutrients' feature and ratios. 
# NOTE: Tree-based models can often learn this, but explicit features can help lighter models.
df['total_nutrients'] = df['N'] + df['P'] + df['K']
df['N_ratio'] = df['N'] / df['total_nutrients']
df['P_ratio'] = df['P'] / df['total_nutrients']
df['K_ratio'] = df['K'] / df['total_nutrients']

# Update numeric features list
numeric_features.extend(['total_nutrients', 'N_ratio', 'P_ratio', 'K_ratio'])

# ==========================================
# 4. Preprocessing Pipeline
# ==========================================
X = df.drop('label', axis=1)
y = df['label']

# Encode Target (Crops -> Numbers)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Create Preprocessing Pipeline
# 1. PowerTransformer: Fixes skewness (Yeo-Johnson) - good for rainfall
# 2. StandardScaler: Standardizes scale (Mean=0, Std=1)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('power', PowerTransformer(method='yeo-johnson')), 
            ('scaler', StandardScaler())
        ]), numeric_features)
    ]
)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# ==========================================
# 5. Model Training & Selection
# ==========================================
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

best_model = None
best_accuracy = 0
results = {}

print("\nTraining Models...")
for name, model in models.items():
    # Create full pipeline: Preprocess -> Model
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"[{name}] Accuracy: {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = clf

print(f"\n🏆 Best Model: {best_model.steps[1][0]} with Accuracy: {best_accuracy:.4f}")

# ==========================================
# 6. Evaluation & Saving
# ==========================================
# Detailed report for the best model
y_pred_best = best_model.predict(X_test)
print("\nClassification Report (Sample):")
# Mapping back numbers to names for the report would be verbose, showing basic metrics
print(classification_report(y_test, y_pred_best, target_names=label_encoder.classes_))

# Save Artifacts
print("\nSaving model and artifacts...")
artifacts = {
    'model': best_model,
    'label_encoder': label_encoder,
    'features': numeric_features # Save feature names to ensure order in app
}
joblib.dump(artifacts, 'crop_recommendation_model.pkl')
print("✅ Model saved as 'crop_recommendation_model.pkl'")