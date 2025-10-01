# ML-Lab

This repository contains all **Machine Learning Lab Practicals (TE7105)** for 7th Semester.  
Each practical includes **problem statement, code implementation, and outputs**.  

## 📌 Practicals List

1. Introduction to ML Tools and Libraries (Iris dataset)  
2. Data Preprocessing and Cleaning (Student Performance dataset)  
3. Principal Component Analysis (Wine dataset)  
4. K-Nearest Neighbors (KNN) Classifier (Iris dataset)  
5. Naïve Bayes Classifier (SMS Spam dataset)  
6. Decision Tree Classifier (PIMA Diabetes dataset)  
7. Random Forest Classifier (PIMA Diabetes dataset)  
8. Simple Linear Regression (Student Scores dataset)  
9. Multiple Linear Regression (Housing Prices dataset)  
10. Logistic Regression (User Purchase dataset)  
11. K-Means Clustering (Mall Customers dataset)  
12. Association Rule Mining (Market Basket dataset)  
13. Artificial Neural Network (MNIST dataset)  
14. Model Evaluation and Tuning (SVM/Random Forest)  
15. Mini Project (choice-based: Heart Disease, Fake News, Stock Price, Weather, etc.)

---

# Practical 1 – Introduction to ML Tools and Libraries (Iris Dataset)

## 🎯 Aim
To understand basic ML tools (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn) and perform basic statistical analysis and visualization on the Iris dataset.

## 📝 Steps
1. Import libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `sklearn.datasets`.
2. Load the **Iris dataset** from scikit-learn.
3. Convert dataset into Pandas DataFrame.
4. Explore dataset with:
   - `head()`, `shape`, `info()`, `describe()`
   - Mean, median, mode, min, max of features
5. Visualizations:
   - Histogram (distribution of features)
   - Pairplot (relationships)
   - Boxplot (spread of values)
6. Summarize findings.

---

# Practical 2 – Data Preprocessing and Cleaning (Student Performance Dataset)

## 🎯 Aim
To perform missing value handling, outlier detection, label encoding, and feature scaling on the Student Performance dataset.

## 📝 Steps
1. Load **student performance dataset** (`student-mat.csv` or `student-por.csv`).
2. Check missing values using `.isnull().sum()`.
3. Handle missing data:
   - Numeric → fill with mean/median
   - Categorical → fill with mode
4. Detect outliers:
   - Boxplots or IQR method
   - Optionally remove outliers
5. Encode categorical features:
   - `LabelEncoder` or `OneHotEncoder`
6. Scale features:
   - `StandardScaler` (standardization) or `MinMaxScaler` (normalization)
7. Confirm dataset is clean and ready for ML.

---

# Practical 3 – Principal Component Analysis (PCA on Wine Dataset)

## 🎯 Aim
To apply PCA on the Wine dataset for dimensionality reduction and visualize the transformed data.

## 📝 Steps
1. Load **Wine dataset** from scikit-learn.
2. Standardize features using `StandardScaler`.
3. Apply PCA with `n_components=2`.
4. Check explained variance ratio.
5. Plot scatter graph of **PC1 vs PC2** with class labels.
6. Interpret PCA results for dimensionality reduction.

---

# Practical 4 – K-Nearest Neighbors (KNN on Iris Dataset)

## 🎯 Aim
To classify Iris flowers using the KNN algorithm, tune K value, and evaluate accuracy with confusion matrix.

## 📝 Steps
1. Load **Iris dataset**.
2. Split data into train/test (70%-30%).
3. Standardize features with `StandardScaler`.
4. Train **KNN classifier** for multiple K values (1–20).
5. Evaluate accuracy for each K.
6. Plot **Accuracy vs K graph**.
7. Choose best K and evaluate:
   - Confusion Matrix
   - Classification Report
8. Plot decision boundaries (using 2 features).

---

# Practical 5 – Naïve Bayes Classifier (SMS Spam Detection)

## 🎯 Aim
To classify SMS messages as spam or ham using the Naïve Bayes algorithm.

## 📝 Steps
1. Load the **SMS Spam Collection dataset**.
2. Preprocess text:
   - Lowercase conversion
   - Remove stopwords/punctuation
   - Convert to numeric using **CountVectorizer / TF-IDF**
3. Split data into train/test sets.
4. Train **MultinomialNB** model.
5. Predict on test set.
6. Evaluate with:
   - Accuracy
   - Precision, Recall, F1-score
   - Confusion Matrix
7. Test with custom SMS input for spam/ham detection.

---
