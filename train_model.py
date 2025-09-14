# train_model.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# Load dataset
df = pd.read_csv("single_genre_artists.csv")

# Only audio features
audio_features = ['danceability', 'energy', 'loudness', 'speechiness',
                  'acousticness', 'instrumentalness', 'liveness',
                  'valence', 'tempo', 'duration_ms']
df_audio = df[audio_features]

# Scale features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_audio)

# Train KMeans
kmeans = KMeans(n_clusters=5, random_state=0, n_init='auto', max_iter=3000)
kmeans.fit(df_scaled)

# Save models
joblib.dump(scaler, "scaler.joblib")
joblib.dump(kmeans, "kmeans_model.joblib")
print("Models saved: scaler.joblib & kmeans_model.joblib")
