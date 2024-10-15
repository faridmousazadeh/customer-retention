# customer_retention_analysis.py

"""
Customer Retention Analysis

This script performs customer segmentation using K-Means clustering to identify customer groups 
for targeted marketing strategies. It includes data preprocessing, clustering, and visualization of segments.

Assumptions: 
- The dataset has features relevant to customer behavior, such as purchase history and demographics.
- A mock dataset will be generated for demonstration purposes.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Generate a mock dataset
np.random.seed(42)
data_size = 500
data = pd.DataFrame({
    'feature1': np.random.rand(data_size) * 100,  # Example feature (e.g., spend)
    'feature2': np.random.rand(data_size) * 50,   # Another feature (e.g., frequency)
})

# Data Preprocessing
data.fillna(method='ffill', inplace=True)  # Forward fill any missing values

# Normalize features for K-Means
features_normalized = (data - data.mean()) / data.std()

# Determine the optimal number of clusters using the Elbow method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(features_normalized)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid()
plt.show()

# Choose the optimal number of clusters (e.g., 3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['cluster'] = kmeans.fit_predict(features_normalized)

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='feature1', y='feature2', hue='cluster', palette='viridis')
plt.title('Customer Segmentation')
plt.xlabel('Feature 1')  # Replace with actual feature names
plt.ylabel('Feature 2')  # Replace with actual feature names
plt.grid()
plt.show()
