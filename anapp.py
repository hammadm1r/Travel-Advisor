from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

# Load and preprocess data
df = pd.read_csv("Tourist Destinations.csv")

# Label Encode category and district
le_category = LabelEncoder()
le_district = LabelEncoder()

df['category_encoded'] = le_category.fit_transform(df['category'])
df['district_encoded'] = le_district.fit_transform(df['district'])

# Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate text embeddings for the description
df['desc_embedding'] = df['Desc'].apply(lambda x: model.encode(str(x)))

# Combine embeddings with encoded labels to form a complete feature vector
def build_feature_vector(row):
    return np.concatenate([
        row['desc_embedding'],
        [row['category_encoded'], row['district_encoded']]
    ])

df['feature_vector'] = df.apply(build_feature_vector, axis=1)
X = np.stack(df['feature_vector'].values)

# Fit Nearest Neighbors model using cosine similarity
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(X)

@app.route('/recommend', methods=['GET'])
def recommend():
    district = request.args.get('district')
    category = request.args.get('category')
    query_desc = request.args.get('desc', '')  # optional text description

    # Basic input validation
    if category and category not in le_category.classes_:
        return jsonify({"error": "Invalid category"}), 400
    if district and district not in le_district.classes_:
        return jsonify({"error": "Invalid district"}), 400

    try:
        desc_vector = model.encode(query_desc)
    except Exception as e:
        return jsonify({"error": "Invalid description"}), 400

    category_encoded = le_category.transform([category])[0] if category else 0
    district_encoded = le_district.transform([district])[0] if district else 0

    # Build user vector
    user_vector = np.concatenate([desc_vector, [category_encoded, district_encoded]])

    # Get nearest neighbors
    distances, indices = knn.kneighbors([user_vector])

    # Prepare response
    recommendations = df.iloc[indices[0]][[
        '_key', 'Desc', 'category', 'district', 'latitude', 'longitude'
    ]].to_dict(orient='records')

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)