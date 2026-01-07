from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load trained Gradient Boosting model (trained on 3 features)
model = joblib.load("gbr_eq_frequency_model.pkl")


@app.route('/predict-frequency', methods=['POST'])
def predict_frequency():
    data = request.json

    min_lat = data['min_lat']
    max_lat = data['max_lat']
    min_lon = data['min_lon']
    max_lon = data['max_lon']
    min_mag = data['min_mag']

    # Limit time window to avoid huge downloads (important)
    start_year = 2010
    end_year = 2024

    url = (
        "https://earthquake.usgs.gov/fdsnws/event/1/query.csv?"
        f"starttime={start_year}-01-01&endtime={end_year}-12-31"
        f"&minlatitude={min_lat}&maxlatitude={max_lat}"
        f"&minlongitude={min_lon}&maxlongitude={max_lon}"
        f"&minmagnitude={min_mag}"
    )

    # Fetch data safely
    try:
        df = pd.read_csv(url)
    except Exception:
        return jsonify({
            "estimated_monthly_frequency": None,
            "note": "Data request failed. Please try a smaller area."
        })

    if df.empty:
        return jsonify({
            "estimated_monthly_frequency": 0,
            "note": "No recorded earthquakes in selected region"
        })

    # Time processing
    df['time'] = pd.to_datetime(df['time'])
    df['month'] = df['time'].dt.to_period('M')

    # Monthly aggregation
    monthly = df.groupby('month').size().reset_index(name='eq_count')

    # If too little data, return historical mean
    if len(monthly) < 6:
        mean_freq = monthly['eq_count'].mean()
        return jsonify({
            "estimated_monthly_frequency": round(float(mean_freq), 2),
            "note": "Estimated using historical average (limited data)"
        })

    # Feature engineering (MATCHES TRAINING FEATURES)
    monthly['lag_1'] = monthly['eq_count'].shift(1)
    monthly['lag_2'] = monthly['eq_count'].shift(2)
    monthly['rolling_mean_3'] = monthly['eq_count'].rolling(3).mean()
    monthly.dropna(inplace=True)

    # Use most recent regional behavior
    recent = monthly.tail(1)

    X_input = [[
        recent['lag_1'].values[0],
        recent['lag_2'].values[0],
        recent['rolling_mean_3'].values[0]
    ]]

    # Predict
    prediction = model.predict(X_input)[0]
    # Regional scaling factor
    regional_mean = monthly['eq_count'].mean()
    national_mean = 20.69
    scaling_factor = regional_mean / national_mean
    adjusted_prediction = prediction * scaling_factor

    return jsonify({
    "estimated_monthly_frequency": round(float(adjusted_prediction), 2),
    "note": "Adjusted using regional seismic intensity"
})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
