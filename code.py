# Machine Learning :  Clusters 
# Importing the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Load dataset

df= pd.read_csv(r"C:\Users\DELL\Downloads\IRIS.csv")

#print the top 5 rows
print("Top 5 rows are : \n",df.head())

#Print the info of dataset
print("The dataset info : \n")
df.info()

# columns in the dataset
df.columns

# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle missing values
print(f"\nMissing values before cleaning:\n{df.isnull().sum()}")

# Fill missing numerical values with column mean
df.fillna(df.mean(numeric_only=True), inplace=True)

print(f"\nMissing values after cleaning:\n{df.isnull().sum()}")

# select relevent columns 
features= df[['sepal_length','sepal_width','petal_length']]

#Standardize the data
Scaler=StandardScaler()
Scaled_features = Scaler.fit_transform(features)

#display the first 10 rows of scaled_features
print(" The first 10 rows of Scaled_features : \n ")
print(Scaled_features[:10])


# Elbow method to find the optimal number of clusters

inertia = []
k_range = range(1,11)

for k in k_range:
    kmeans=KMeans(n_clusters= k , random_state= 42)
    kmeans.fit(Scaled_features)
    inertia.append(kmeans.inertia_)


# plot the elbow graph

plt.figure(figsize=(12, 5))
plt.plot(k_range ,inertia ,marker ='o')
plt.title("Elbow method to find optimal clusters  k ")
plt.xlabel("Number of cluster(k)")
plt.ylabel("Inertia")
plt.show()

# perform kmeans clustering with optimal k (from elbow method assume k=3 )
optimal_k = 3
kmeans= KMeans(n_clusters = optimal_k ,random_state=42)
cluster_labels = kmeans.fit_predict(Scaled_features)

# add a feature "clusters" to the main dataset
df['Clusters'] = cluster_labels

# print the top 5 rows of updated dataset
print("\n Top 5 rows of updated dataset :\n")
print(df.head())

# Visualize the Clusters

plt.figure(figsize=(8,4))
sns.scatterplot(x=Scaled_features[:,0],y=Scaled_features[:,1],hue=cluster_labels,palette='viridis',s=80,alpha=0.8)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='red',label='centroids')
plt.title('customer segment ')
plt.xlabel('Feature 1 (scaled) ')
plt.ylabel('Feature 2 (scaled) ') 
plt.legend()
plt.show()
