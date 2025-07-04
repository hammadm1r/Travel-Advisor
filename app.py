from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# Load Data
df = pd.read_csv("Tourist Destinations.csv")

# Preprocess
le_category = LabelEncoder()
le_district = LabelEncoder()

df['category_encoded'] = le_category.fit_transform(df['category'])
df['district_encoded'] = le_district.fit_transform(df['district'])

# For Lat-Long KNN
geo_knn = NearestNeighbors(n_neighbors=1, metric='euclidean')  # 1 closest location
geo_knn.fit(df[['latitude', 'longitude']])

@app.route('/recommend', methods=['GET'])
def recommend():
    district = request.args.get('district')
    category = request.args.get('category')

    # Validate category and district
    if district and district not in le_district.classes_:
        return jsonify({"error": "Invalid district"}), 400
    if category and category not in le_category.classes_:
        return jsonify({"error": "Invalid category"}), 400

    # Filter the data based on the category and district
    filtered_df = df.copy()
    if category:
        filtered_df = filtered_df[filtered_df['category'] == category]
    if district:
        filtered_df = filtered_df[filtered_df['district'] == district]

    if filtered_df.empty:
        return jsonify({"message": "No recommendations found"}), 404

    filtered_df['category_encoded'] = le_category.transform(filtered_df['category'])
    filtered_df['district_encoded'] = le_district.transform(filtered_df['district'])

    n_neighbors = min(5, len(filtered_df))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(filtered_df[['category_encoded', 'district_encoded']])

    user_pref = [[le_category.transform([category])[0], le_district.transform([district])[0]]] if category and district else []

    if user_pref:
        distances, indices = knn.kneighbors(user_pref)
        recommendations = filtered_df.iloc[indices[0]][[
            '_key', 'Desc', 'category', 'district', 'latitude', 'longitude'
        ]].to_dict(orient='records')
    else:
        recommendations = filtered_df[[
            '_key', 'Desc', 'category', 'district', 'latitude', 'longitude'
        ]].to_dict(orient='records')

    return jsonify(recommendations)

@app.route('/recommend/geo', methods=['GET'])
def recommend_by_geo():
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing latitude/longitude"}), 400

    # Get nearest destination
    _, indices = geo_knn.kneighbors([[lat, lon]])
    nearest = df.iloc[indices[0][0]][[
        '_key', 'Desc', 'category', 'district', 'latitude', 'longitude'
    ]].to_dict()

    return jsonify(nearest)

# if __name__ == '__main__':
#     app.run(debug=True)

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # use PORT from env
    app.run(debug=False, host='0.0.0.0', port=port)


