import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def clustK(data):

    kmeans_default = KMeans()
    kmeans_default.fit(data)


    wcss_default = kmeans_default.inertia_
    print(f'WCSS for default k (number of clusters): {wcss_default:.2f}')

    silhouette_score_default = silhouette_score(data, kmeans_default.labels_)
    print(f'Silhouette Score for default k: {silhouette_score_default:.2f}')

    return kmeans_default, wcss_default, silhouette_score_default

def plott(data, kmeans_model, title):

    plt.scatter(data['Individuals_Affected'], data['year'], c=kmeans_model.labels_, cmap='rainbow')
    plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], s=100, c='black', label='Centroids')
    plt.xlabel('Individuals Affected')
    plt.ylabel('YeclustKplotar')
    plt.legend()
    plt.title(title)
    plt.show()

if __name__ == "__main__":

    data = pd.read_csv('Cyber Security Breaches.csv')


    selected_data = data[['Individuals_Affected', 'year']]

    kmeans_default, wcss_default, silhouette_score_default = clustK(selected_data)


    plott(selected_data, kmeans_default, 'Data Points and Cluster Centroids for Default k')


    k_values = range(2, 21)

    wcss_values = []
    silhouette_scores = []


    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(selected_data)

        wcss = kmeans.inertia_
        wcss_values.append(wcss)

        silhouette = silhouette_score(selected_data, kmeans.labels_)
        silhouette_scores.append(silhouette)

        print(f'WCSS for k={k}: {wcss:.2f}, Silhouette Score for k={k}: {silhouette:.2f}')

    
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

