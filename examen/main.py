import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import streamlit as st

# Încărcarea setului de date
st.title("Clustering pe setul de date Breast Cancer Wisconsin - bilet 20")

# Citirea datelor
file_path = "C:\\Users\\Vladut\\Desktop\\examen\\breast+cancer\\breast-cancer.data"

column_names = ["Class", "Age", "Menopause", "Tumor_Size", "Inv_Nodes", "Node_Caps", "Deg_Malig", "Breast", "Breast_Quad", "Irradiat"]
df = pd.read_csv(file_path, names=column_names)

# Afisare date brute
st.subheader("Date brute")
st.dataframe(df.head())

# Înlocuirea valorilor '?' cu NaN
df.replace("?", np.nan, inplace=True)

# Gestionarea valorilor lipsă
df["Node_Caps"].fillna(df["Node_Caps"].mode()[0], inplace=True)
df["Breast_Quad"].fillna(df["Breast_Quad"].mode()[0], inplace=True)

# Transformarea variabilelor categoriale
categorical_cols = ["Age", "Menopause", "Tumor_Size", "Inv_Nodes", "Node_Caps", "Breast", "Breast_Quad", "Irradiat"]
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

df["Class"] = LabelEncoder().fit_transform(df["Class"])

# Normalizarea datelor
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

# Afisare date preprocesate
st.subheader("Date normalizate")
st.dataframe(df_scaled.head())

# Determinarea numărului optim de clustere prin metoda Elbow
inertia = []
cluster_range = range(1, 11)
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled.iloc[:, 1:])
    inertia.append(kmeans.inertia_)

# Afișarea graficului "Elbow Method"
st.subheader("Metoda Elbow pentru K-Means")
fig, ax = plt.subplots()
ax.plot(cluster_range, inertia, marker='o', linestyle='-')
ax.set_xlabel("Numărul de clustere (k)")
ax.set_ylabel("Inertia (SSD)")
ax.set_title("Elbow Method")
st.pyplot(fig)

# Aplicarea K-Means cu k=3 (valoare optimă observată din grafic)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_scaled["KMeans_Cluster"] = kmeans.fit_predict(df_scaled.iloc[:, 1:])

# Aplicarea clusteringului ierarhic aglomerativ
agg_cluster = AgglomerativeClustering(n_clusters=3)
df_scaled["Agg_Cluster"] = agg_cluster.fit_predict(df_scaled.iloc[:, 1:])

# Evaluarea clusteringului folosind coeficientul Silhouette
silhouette_kmeans = silhouette_score(df_scaled.iloc[:, 1:-2], df_scaled["KMeans_Cluster"])
silhouette_agg = silhouette_score(df_scaled.iloc[:, 1:-2], df_scaled["Agg_Cluster"])

# Afișare scor Silhouette
st.subheader("Evaluare Silhouette")
st.write(f"Silhouette Score pentru K-Means: {silhouette_kmeans:.4f}")
st.write(f"Silhouette Score pentru Clustering Ierarhic: {silhouette_agg:.4f}")

# Vizualizarea clusteringului
st.subheader("Distribuția clusterelor")
fig_clusters, ax_clusters = plt.subplots()
sns.scatterplot(x=df_scaled.iloc[:, 1], y=df_scaled.iloc[:, 2], hue=df_scaled["KMeans_Cluster"], palette='viridis', ax=ax_clusters)
ax_clusters.set_title("Clustering cu K-Means")
st.pyplot(fig_clusters)

# Compararea clusterelor cu eticheta originală
st.subheader("Distribuția clusterelor vs. Eticheta originală")
fig_compare, ax_compare = plt.subplots()
sns.countplot(x=df_scaled["KMeans_Cluster"], hue=df["Class"], palette='coolwarm', ax=ax_compare)
ax_compare.set_title("Distribuția etichetelor în funcție de cluster")
st.pyplot(fig_compare)

if __name__ == "__main__":
    st.write("Aplicația rulează! Deschide localhost pentru vizualizare.")
