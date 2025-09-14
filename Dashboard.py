# Dashboard.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import feature_extractor

st.title("🎵 MP3 Music Genre Predictor")

# Upload MP3
uploaded_file = st.file_uploader("Upload MP3 file", type="mp3")
if uploaded_file:
    temp_file = "temp_song.mp3"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract audio features
    df_song = feature_extractor.extract_features(temp_file)

    # Load trained scaler and KMeans
    scaler = joblib.load("scaler.joblib")
    kmeans = joblib.load("kmeans_model.joblib")

    # Audio features in same order as training
    audio_features = ['danceability', 'energy', 'loudness', 'speechiness',
                      'acousticness', 'instrumentalness', 'liveness',
                      'valence', 'tempo', 'duration_ms']

    # Ensure all columns exist and are in correct order
    for col in audio_features:
        if col not in df_song.columns:
            df_song[col] = 0  # fill missing
    df_song = df_song[audio_features]

    # Scale features
    df_scaled = scaler.transform(df_song)

    # Predict cluster
    cluster = kmeans.predict(df_scaled)[0]

    # Map cluster to genre
    cluster_labels = {
        0: 'Instrumental / Acoustic',
        1: 'Mainstream / Party',
        2: 'Happy / Dance',
        3: 'Vocal / Speech-heavy',
        4: 'Chill / Relaxing'
    }

    st.success(f"Predicted Genre: {cluster_labels[cluster]}")
