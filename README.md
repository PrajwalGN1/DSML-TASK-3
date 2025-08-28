# DSML-TASK-3
#  K-Means Clustering on Iris Dataset  

This project demonstrates **K-Means clustering** on the famous **Iris dataset**, one of the most well-known datasets in machine learning. The goal is to group iris flowers into clusters based on their **sepal** and **petal** dimensions, without using the species labels.  

## 📊 Dataset
- **Name:** Iris Dataset  
- **Features:**
  - Sepal Length  
  - Sepal Width  
  - Petal Length  
  - Petal Width  
- **Target (not used in clustering):** Iris Setosa, Iris Versicolor, Iris Virginica  

##  Steps Performed
1. **Data Preprocessing**  
   - Loaded the Iris dataset  
   - Standardized features for better clustering performance  

2. **Choosing Optimal K (Clusters)**  
   - Used the **Elbow Method** to find the best value of k  
   - Found optimal k = 3  

3. **K-Means Clustering**  
   - Applied K-Means with k = 3  
   - Assigned cluster labels to each flower  

4. **Visualization**  
   - Plotted clusters in 2D using petal/sepal features  
   - Compared clustering with actual species  

## 📈 Results
- The algorithm grouped flowers into 3 distinct clusters.  
- Clusters largely matched the actual species labels (Setosa, Versicolor, Virginica).  


## 🛠️ Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  


## 🚀 How to Run

# Clone the repository
git clone https://github.com/your-username/iris-kmeans-clustering.git

# Navigate to folder
cd iris-kmeans-clustering

# Install requirements
pip install -r requirements.txt

# Run the notebook/script
python kmeans_iris.py '''

