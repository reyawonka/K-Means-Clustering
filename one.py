import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def clustK(data, k_values):
    wcss_values = []
    silhouette_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)

        wcss = kmeans.inertia_
        wcss_values.append(wcss)

        silhouette = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(silhouette)

        print(f' k={k}:for WCSS  {wcss:.2f}, k={k}:for  Silhouette Score {silhouette:.2f}')

    return wcss_values, silhouette_scores

def optK(wcss_values):
    reduction_in_wcss = [wcss_values[i] - wcss_values[i-1] for i in range(1, len(wcss_values))]
    optimal_k = 2 + reduction_in_wcss.index(max(reduction_in_wcss))
    return optimal_k

if __name__ == "__main__":
    np.random.seed(0)
    random_data = np.random.rand(100, 2)
    k_values = range(2, 21)

    wcss_values, silhouette_scores = clustK(random_data, k_values)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_values, wcss_values, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.title('WCSS vs. Number of Clusters')

    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')

    plt.tight_layout()
    plt.show()

    optimal_k = optK(wcss_values)
    print(f'{optimal_k}: is the optimal number of clusters (k) based on WCSS')
    
    
    optimal_kmeans = KMeans(n_clusters=optimal_k)
    optimal_kmeans.fit(random_data)

    plt.scatter(random_data[:, 0], random_data[:, 1], c=optimal_kmeans.labels_, cmap='rainbow')
    plt.scatter(optimal_kmeans.cluster_centers_[:, 0], optimal_kmeans.cluster_centers_[:, 1], s=100, c='black', label='Centroids')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title(f'Data Points and Cluster Centroids for Optimal k ({optimal_k})')
    plt.show()

