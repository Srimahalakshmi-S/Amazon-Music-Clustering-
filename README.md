# Amazon-Music-Clustering-

🎵 Amazon Music Clustering

Automatically cluster Amazon Music songs based on audio features using unsupervised learning. This helps in playlist curation, song recommendations, and artist/market insights.

🚀 Project Overview

With millions of songs available, manually grouping tracks by genre or mood is impractical. This project uses K-Means clustering to group songs with similar audio characteristics such as danceability, energy, loudness, tempo, and valence.

Key goals:

Generate meaningful song clusters

Visualize clusters for interpretability

Provide actionable insights for music recommendation and analysis

🗂 Dataset

File: single_genre_artists.csv

Numeric Features:
danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, popularity_songs, followers, popularity_artists

Reference Columns:
track_id, track_name, artist_name, genres

The dataset describes how a song “sounds” in terms of rhythm, mood, intensity, and instrumentation.

🛠 Approach

Data Cleaning & Preprocessing:

Drop irrelevant columns (track_id, track_name, artist_name, explicit, release_date, genres, mode, key, time_signature)

Handle missing values

Feature Scaling:

Normalize numeric features using StandardScaler

Clustering:

K-Means applied on scaled data

Determine optimal number of clusters with Elbow Method and Silhouette Score

Cluster Evaluation:

Silhouette Score for cohesion and separation

Mean feature analysis per cluster to interpret “mood” or style

Visualization:

PCA scatterplots for 2D representation

Heatmaps and bar plots for feature comparisons

Export:

Add cluster labels to dataset

Save final CSV with clusters

📊 Results

Songs grouped into meaningful clusters reflecting audio similarities

Identified cluster profiles (e.g., high danceability → “Party Tracks”, high acousticness → “Chill Acoustic”)

Clusters can be used for playlist generation, music recommendation, and artist analysis

⚡ Evaluation Metrics
Metric	Purpose
Silhouette Score	Measures similarity within clusters vs others
Davies-Bouldin Index	Measures intra-cluster similarity; lower = better
Inertia (WCSS)	K-Means cluster compactness
Cluster Balance	Checks distribution across clusters
Feature Interpretability	Understand dominant audio features per cluster
🛠 Tech Stack

Python | Pandas | NumPy | scikit-learn | Matplotlib | Seaborn

📂 Deliverables

Source Code: Jupyter Notebook or Python script

Clustered Dataset: CSV with cluster labels

Visualizations: PCA scatterplots, feature heatmaps, distribution plots

Optional: Streamlit app for interactive cluster exploration
