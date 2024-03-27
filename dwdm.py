import streamlit as st
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples
import numpy as np

# Function to perform clustering
def perform_clustering(num_samples):
    # Generate sample data
    X, _ = make_blobs(n_samples=num_samples, centers=3, random_state=42)

    # Fit AGNES clustering model
    agnes = AgglomerativeClustering(n_clusters=3)
    labels = agnes.fit_predict(X)

    # Display cluster assignments
    st.write("Cluster Assignments:")
    st.write(labels)

    # Calculate and print silhouette coefficient for each data point
    silhouette_values = silhouette_samples(X, labels)
    for i, s in enumerate(silhouette_values):
        st.write(f"Data point {i}: Silhouette coefficient = {s}")
        if s > 0:
            st.write("This data point is well clustered.")
        elif s == 0:
            st.write("This data point is on the boundary between clusters.")
        else:
            st.write("This data point may be assigned to the wrong cluster.")
        st.write()

# Streamlit app
def main():
    st.title("AGNES Clustering with Streamlit")
    num_samples = st.number_input("Enter the number of samples in the dataset:", min_value=10, max_value=1000, step=10, value=10)
    perform_clustering(num_samples)

if __name__ == "__main__":
    main()
