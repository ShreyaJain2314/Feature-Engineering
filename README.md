# Feature-Engineering

#TITANIC DATASET

The code shows a data preprocessing and logistic regression modeling pipeline for the Titanic dataset, which is a common dataset used for binary classification tasks (predicting survival or not survival in the Titanic disaster). Let's break down what's happening in the code step by step:

Data Loading: The code starts by loading two datasets: train and test from CSV files. These datasets typically contain information about passengers on the Titanic, with the train dataset having the target variable 'Survived' (1 for survived, 0 for not survived).

Data Exploration and Visualization:Various data exploration and visualization tasks are performed:
    Checking the counts of values in the 'Cabin' column.
    Filling missing values in the 'Age' column with a placeholder value (-0.5).
    Creating bar plots using Seaborn to visualize relationships between features and the target variable 'Survived' (e.g., gender, passenger class, age group).
    
Feature Engineering: 
    Age is binned into groups (e.g., 'baby', 'child', 'adult') using the pd.cut function.
    Columns 'Ticket', 'Fare', and 'Cabin' are dropped from the dataset as they are not used for modeling.
    Missing values in the 'Embarked' column are filled with 'S'.

Creating Title Feature:
    Titles are extracted from the 'Name' column using regular expressions.
    Titles are then categorized into common groups (e.g., 'Mr', 'Miss', 'Mrs') and rare titles are combined into 'Rare'.
    Some titles like 'Mlle' and 'Mme' are mapped to 'Miss' and 'Mrs', respectively.
    The 'Title' feature is then mapped to numeric values.
    
Imputing Missing Age Values Based on Title: The code attempts to impute missing age values based on the mode (most frequent) age group for each title category.

Label Encoding: Label encoding is applied to categorical features ('Sex', 'AgeGroup', 'Embarked') using LabelEncoder from scikit-learn.

Model Training and Prediction: A logistic regression model (LogisticRegression) is trained on the preprocessed training data (xtrain and ytrain).

The model is used to make predictions on the test data (xtest).The predictions are stored in a DataFrame along with the 'PassengerId'.


# PCA Implementation

The code is performing several data preprocessing and visualization steps, mainly focused on K-means clustering and Principal Component Analysis (PCA) visualization of the data. Let's break down what's happening in the code step by step:

Import necessary libraries: matplotlib.pyplot for plotting. seaborn for enhancing the appearance of plots. pandas for data manipulation. Also imported data from a CSV file named "global_heat_index.csv" using pd.read_csv() and store it in a DataFrame called df.

Data Cleaning: Two subsets of data are created, clean1 and clean2, based on conditions where the 'Hour' column is less than 6 or greater than 18, respectively.The code prints messages indicating the data cleaning process.

Further Data Cleaning: The original DataFrame df is modified to remove rows where the 'Hour' column is either less than 6 or greater than 18 using boolean indexing.

Data Splitting and Standardization: The data is split into features x and target y. train_test_split from sklearn is used to split the data into training and testing sets. Standardization is performed on both the feature data (x) and the target data (y) using StandardScaler.

K-Means Clustering: KMeans clustering is applied with 6 clusters (n_clusters=6) to the standardized training data x_train. Cluster labels are assigned to each data point and stored in the labels variable. The cluster centers are also calculated and stored in the cluster_centers variable.

PCA Transformation: Principal Component Analysis (PCA) is applied to reduce the dimensionality of the data to 2 components. A scatter plot is created for each data point in the reduced feature space (pca) based on their K-means cluster label. Different colors and markers are used for each cluster, and the points are stored in various variables (cl_6, c2_6, ..., c6_6) based on the cluster label.

Visualization: The code sets the figure size and adds a legend for different clusters.

Finally, it displays the PCA scatter plot with different clusters colored and marked according to their assigned labels.


#correlation matrix

Importing Libraries:

The code begins by importing the necessary libraries: matplotlib.pyplot for plotting. seaborn for enhancing the appearance of plots. pandas for data manipulation. Data is read from a CSV file named "global_heat_index.csv" using pd.read_csv() and stored in a DataFrame called df.

Data Cleaning: Two subsets of data are created based on conditions:
clean1: Rows where the 'Hour' column is less than 6.
clean2: Rows where the 'Hour' column is greater than 18.
The code prints messages indicating the data cleaning process.
However, the results of these data subsets (clean1 and clean2) are not displayed or used further in the code.

Further Data Cleaning: The original DataFrame df is modified using boolean indexing to remove rows where the 'Hour' column is either less than 6 or greater than 18.

Data Preprocessing: Features and target variables (x and y) are extracted from the DataFrame. A MinMaxScaler is applied to scale the feature data (x) to a range of [0, 1]. This scaling is important, especially when working with machine learning algorithms that are sensitive to feature scales.

Correlation Matrix and Heatmap: A correlation matrix is calculated using df.corr(). This matrix shows the pairwise correlations between numerical columns in the DataFrame.
A heatmap is created using Seaborn (sns.heatmap()) to visualize the correlation matrix. The annot=True parameter adds numeric annotations to the cells to display the correlation coefficients.

Visualization: The heatmap is displayed using Matplotlib's plt.figure() and plt.show() functions. The heatmap visually represents how different columns in the DataFrame are correlated with each other. Warmer colors (e.g., closer to 1) indicate stronger positive correlations, while cooler colors (e.g., closer to -1) indicate stronger negative correlations.

