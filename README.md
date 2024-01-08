# K Means Clustering

<!DOCTYPE html>
<html>
<head>


</head>
<body>

<h1>Lab on K Means Clustering</h1>
<p>This repository contains the code and documentation for a lab exercise focused on K Means Clustering, a popular unsupervised machine learning algorithm.</p>

<h2>Overview of Scripts</h2>

<h3>1. ClusterEvaluation.py </h3>
<p>Performs clustering with different values of k and evaluates the performance using metrics like WCSS and silhouette scores.</p>
<pre class="code">
# Sample Code Snippet
for k in k_values:
    kmeans = KMeans(n_clusters=k)
    #...
    silhouette = silhouette_score(data, kmeans.labels_)
    #...
</pre>

<h3>2. DefaultClusterAnalysis.py </h3>
<p>Implements K Means clustering with a default set of parameters and calculates the WCSS and silhouette score for assessment.</p>
<pre class="code">
# Sample Code Snippet
kmeans_default = KMeans()
#...
print(f'Silhouette Score for default k: {silhouette_score_default:.2}')
#...
</pre>

<h2>Conclusion</h2>
<p>This lab highlights the significance of selecting the right number of clusters in K Means and how different metrics can be used to evaluate the performance of the clustering algorithm.</p>

</body>
</html>
