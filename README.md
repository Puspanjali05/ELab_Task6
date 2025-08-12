# ELab_Task6

# 📍 K-Nearest Neighbors (KNN) - Classification

This project is a part of the AI & ML Internship program.  
The objective is to implement a **K-Nearest Neighbors (KNN)** classifier, tune the hyperparameter **K**, evaluate performance, and visualize decision boundaries.

---

## 📌 Objective

Train and evaluate a KNN classification model to:
- Predict target classes based on input features.
- Understand the effect of different **K** values on model performance.
- Apply feature normalization for distance-based learning.
- Visualize decision boundaries for interpretation.

---

## 🛠️ Tools & Libraries Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Joblib (for saving models)

---

## 📂 Dataset

The dataset used is the **[Iris Dataset](https://www.kaggle.com/datasets/uciml/iris)**.  
**Target Variable:**  
- `0` → Setosa  
- `1` → Versicolor  
- `2` → Virginica  

**Features:**  
- Sepal length (cm)  
- Sepal width (cm)  
- Petal length (cm)  
- Petal width (cm)  

---

## 📊 Project Process

### 1. Data Loading & Preprocessing
- Loaded dataset using pandas.
- Checked shape, columns, and missing values.
- Normalized features using **StandardScaler** to improve distance calculations.
- Split data into 80% train and 20% test sets with stratification.

### 2. Baseline KNN Model
- Trained `KNeighborsClassifier` with `n_neighbors=5`.
- Evaluated using accuracy, confusion matrix, and classification report.

### 3. Hyperparameter Tuning
- Tried multiple values of **K** (1 to 30, odd numbers preferred).
- Compared **uniform** vs **distance** weighting.
- Tested **Euclidean (p=2)** and **Manhattan (p=1)** distances.
- Used **GridSearchCV** for best parameters.

### 4. Evaluation
- Chose the model with the highest cross-validation accuracy.
- Tested the best model on the hold-out test set.
- Generated metrics: Accuracy, Precision, Recall, F1-score.

### 5. Visualization
- Plotted confusion matrix.
- Visualized decision boundaries in 2D using PCA.
- Showed effect of different **K** values on accuracy.

---

## 📈 Key Insights
- Feature scaling significantly improved KNN accuracy.
- Smaller **K** captured finer patterns but was sensitive to noise.
- Larger **K** provided smoother decision boundaries but risked underfitting.
- Decision boundary plots offered visual clarity on classification regions.

---

## 📁 Files in Repository

| File Name                | Description                                         |
|--------------------------|-----------------------------------------------------|
| `ElevateLabs_Task6.ipynb`| Full code and visualizations for the project        |
| `iris.csv`               | Dataset                                             |
| `README.md`              | Project documentation (this file)                   |

