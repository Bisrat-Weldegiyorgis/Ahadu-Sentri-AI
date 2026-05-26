import pandas as pd
import joblib

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("trained_model.pkl")

feature_columns = joblib.load("feature_columns.pkl")

label_encoder = joblib.load("label_encoder.pkl")

print("✅ Model loaded")

# =====================================================
# SAMPLE LIVE DATA
# =====================================================

sample = {
    " Destination Port": 80,
    " Flow Duration": 100000,
    " Total Fwd Packets": 10,
    " Total Backward Packets": 5,
    "Total Length of Fwd Packets": 500,
    " Total Length of Bwd Packets": 300
}

# =====================================================
# DATAFRAME
# =====================================================

df = pd.DataFrame([sample])

# =====================================================
# MATCH TRAINING FEATURES
# =====================================================

df = pd.get_dummies(df)

df = df.reindex(
    columns=feature_columns,
    fill_value=0
)

# =====================================================
# PREDICT
# =====================================================

prediction = model.predict(df)

label = label_encoder.inverse_transform(prediction)

print("\\nPrediction:", label[0])
