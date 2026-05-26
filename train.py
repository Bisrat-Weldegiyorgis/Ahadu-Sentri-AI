import pandas as pd
import glob
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =====================================================
# LOAD ALL DATASETS
# =====================================================

print("Loading dataset files...")

files = glob.glob("datasets/*.csv")

df_list = []

for file in files:
    print("Loading:", file)
    temp = pd.read_csv(file, low_memory=False)
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)

print("Total rows before cleaning:", len(df))

# =====================================================
# CLEAN COLUMNS
# =====================================================

df.columns = df.columns.str.strip()

# =====================================================
# FIND LABEL COLUMN
# =====================================================

label_candidates = ["Label", "label", "Class", "class"]

label_col = None
for col in label_candidates:
    if col in df.columns:
        label_col = col
        break

if label_col is None:
    raise Exception("No label column found!")

print("Using label column:", label_col)

# =====================================================
# CLEAN DATA (VERY IMPORTANT)
# =====================================================

df = df.replace([float("inf"), -float("inf")], 0)
df = df.fillna(0)

# =====================================================
# BINARY LABEL
# =====================================================

df[label_col] = df[label_col].apply(
    lambda x: 0 if str(x).strip().lower() in ["benign", "normal"] else 1
)

# =====================================================
# SPLIT FEATURES / LABEL
# =====================================================

X = df.drop(columns=[label_col])
y = df[label_col]

# =====================================================
# KEEP ONLY NUMERIC FEATURES
# =====================================================

X = X.select_dtypes(include=["number"])

print("Features shape:", X.shape)

# =====================================================
# 🔥 MEMORY FIX (IMPORTANT)
# =====================================================

print("Sampling dataset to avoid memory crash...")

df_sample_size = min(300000, len(X))

X = X.sample(n=df_sample_size, random_state=42)
y = y.loc[X.index]

print("Reduced dataset size:", len(X))

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# MODEL (LIGHTER = STABLE)
# =====================================================

model = RandomForestClassifier(
    n_estimators=50,
    max_depth=15,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# =====================================================
# TRAIN
# =====================================================

print("Training model...")
model.fit(X_train, y_train)

# =====================================================
# EVALUATION
# =====================================================

pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n")
print(classification_report(y_test, pred))

# =====================================================
# SAVE MODEL + FEATURES
# =====================================================

joblib.dump(model, "ids_model.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")

print("\nModel saved successfully!")
print("Features saved:", len(X.columns))
