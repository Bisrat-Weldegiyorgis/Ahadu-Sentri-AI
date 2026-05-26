import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# =====================================================
# LOAD DATASET
# =====================================================

DATASET_PATH = "datasets/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded:", df.shape)

# =====================================================
# CLEAN COLUMN NAMES
# =====================================================

df.columns = df.columns.str.strip()

# =====================================================
# REMOVE INVALID VALUES
# =====================================================

df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.dropna(inplace=True)

print("After cleaning:", df.shape)

# =====================================================
# REMOVE DUPLICATES
# =====================================================

df.drop_duplicates(inplace=True)

print("After duplicates removed:", df.shape)

# =====================================================
# TARGET LABEL
# =====================================================

TARGET = "Label"

# =====================================================
# FEATURES / LABELS
# =====================================================

X = df.drop(columns=[TARGET])

y = df[TARGET]

# =====================================================
# ENCODE LABELS
# =====================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

# =====================================================
# SAVE LABEL ENCODER
# =====================================================

joblib.dump(label_encoder, "label_encoder.pkl")

# =====================================================
# HANDLE NON-NUMERIC COLUMNS
# =====================================================

X = pd.get_dummies(X)

# =====================================================
# SAVE FEATURE COLUMNS
# =====================================================

feature_columns = X.columns.tolist()

joblib.dump(feature_columns, "feature_columns.pkl")

print("Total features:", len(feature_columns))

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# =====================================================
# SAVE PROCESSED DATA
# =====================================================

joblib.dump((X_train, X_test, y_train, y_test), "processed_data.pkl")

print("Processed dataset saved successfully!")
