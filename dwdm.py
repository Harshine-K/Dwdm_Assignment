import streamlit as st
import numpy as np

# Function to generate sample data
def generate_sample_data(num_samples):
    np.random.seed(42)
    centers = [[-1, -1], [0, 0], [1, 1]]
    X = np.zeros((num_samples, 2))
    labels = np.zeros(num_samples)
    for i in range(num_samples):
        center_idx = np.random.randint(len(centers))
        X[i] = np.random.randn(2) + centers[center_idx]
        labels[i] = center_idx
    return X, labels

# Function to perform Agglomerative Clustering
def agglomerative_clustering(X, n_clusters):
    num_samples = X.shape[0]
    distances = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            distances[i, j] = np.linalg.norm(X[i] - X[j])
    
    clusters = [{i} for i in range(num_samples)]
    while len(clusters) > n_clusters:
        min_distance = np.inf
        min_i = -1
        min_j = -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = 0
                for idx_i in clusters[i]:
                    for idx_j in clusters[j]:
                        distance += distances[idx_i, idx_j]
                distance /= len(clusters[i]) * len(clusters[j])
                if distance < min_distance:
                    min_distance = distance
                    min_i = i
                    min_j = j
        clusters[min_i] |= clusters[min_j]
        del clusters[min_j]

    labels = np.zeros(num_samples)
    for idx, cluster in enumerate(clusters):
        for i in cluster:
            labels[i] = idx

    return labels

# Function to calculate silhouette coefficients
def silhouette_coefficients(X, labels):
    num_samples = X.shape[0]
    silhouette_values = np.zeros(num_samples)
    for i in range(num_samples):
        a_i = 0
        b_i = np.inf
        for j in range(num_samples):
            if labels[j] == labels[i] and j != i:
                a_i += np.linalg.norm(X[i] - X[j])
            elif labels[j] != labels[i]:
                b_i = min(b_i, np.linalg.norm(X[i] - X[j]))
        a_i /= max(1, np.sum(labels == labels[i]) - 1)
        silhouette_values[i] = (b_i - a_i) / max(a_i, b_i)
    return silhouette_values

# Streamlit app
def main():
    st.title("AGNES Clustering without Predefined Modules")
    num_samples = st.number_input("Enter the number of samples in the dataset:", min_value=10, max_value=1000, step=10, value=10)

    # Generate sample data
    X, true_labels = generate_sample_data(num_samples)

    # Perform Agglomerative Clustering
    n_clusters = 3
    predicted_labels = agglomerative_clustering(X, n_clusters)

    # Display cluster assignments
    st.write("Cluster Assignments:")
    st.write(predicted_labels)

    # Calculate and display silhouette coefficients
    silhouette_values = silhouette_coefficients(X, predicted_labels)
    for i, s in enumerate(silhouette_values):
        st.write(f"Data point {i}: Silhouette coefficient = {s}")
        if s > 0:
            st.write("This data point is well clustered.")
        elif s == 0:
            st.write("This data point is on the boundary between clusters.")
        else:
            st.write("This data point may be assigned to the wrong cluster.")
        st.write()

if __name__ == "__main__":
    main()
