import pandas as pd
import joblib

model = joblib.load("trained_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def predict_flow(flow_features):

    df = pd.DataFrame([flow_features])

    df = pd.get_dummies(df)

    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction = model.predict(df)

    label = label_encoder.inverse_transform(prediction)

    return label[0]
