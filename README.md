# Iris Flower Classification Using Decision Trees

An end-to-end machine learning lab demonstrating multi-class classification on the classic Fisher's Iris dataset using Scikit-Learn's Decision Tree algorithm. This project covers exploratory data analysis (EDA), tree architecture visualization, evaluation metrics, and hyperparameter tuning.

Developed as part of **CMSC 478: Introduction to Machine Learning**.

---

## 📌 Project Overview

The objective of this project is to classify iris flowers into three distinct species based on four morphological measurements using a supervised Decision Tree Classifier.

Key aspects explored in this notebook:
* Data exploration, attribute inspection, and 2D feature scatter plotting.
* Training a Decision Tree with entropy-based information gain (`criterion="entropy"`, `max_depth=3`).
* Visualizing decision tree paths and split logic using `sklearn.tree.plot_tree`.
* Evaluating performance via train/test splitting, confusion matrices, and precision/recall/F1 metrics.
* Experimenting with split ratios and random seeds to achieve optimal test accuracy (up to **98.67%**).

---

## 📊 Dataset

This project utilizes the built-in **Scikit-Learn Iris Plants Dataset**:

* **Samples:** 150 total (50 per class)
* **Features (4 numeric attributes in cm):**
  1. Sepal Length
  2. Sepal Width
  3. Petal Length
  4. Petal Width
* **Classes (3 target species):**
  * `0`: *Iris-setosa*
  * `1`: *Iris-versicolor*
  * `2`: *Iris-virginica*

---

## ⚙️ Workflow & Implementation

1. **Exploratory Data Analysis (EDA):**
   * Inspect dictionary keys, dataset summary (`DESCR`), and array shapes ($150 \times 4$).
   * Tabulate features and labels using Pandas DataFrames.
   * Visualize sepal feature separation across species using Matplotlib scatter plots.

2. **Model Training & Architecture Visualization:**
   * Instantiate `DecisionTreeClassifier` with `criterion="entropy"` and `max_depth=3`.
   * Plot the full decision hierarchy to inspect node splits, entropy reduction, and sample distributions.
   * Run inference on synthetic/modified data points.

3. **Evaluation (60/40 Split):**
   * Split data using `train_test_split` (60% training, 40% testing).
   * Evaluate test predictions using `ConfusionMatrixDisplay` and `classification_report`.
   * Baseline Test Accuracy: **95.00%**.

4. **Optimization Exercise:**
   * Tune dataset partitioning (`test_size=0.50`, `random_state=42`) and model randomness.
   * Optimized Test Accuracy: **98.67%**.

---

## 📈 Performance Summary

### Baseline Model (60/40 Split)
* **Overall Accuracy:** `95.00%`

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Setosa (0)** | 1.00 | 1.00 | 1.00 | 16 |
| **Versicolor (1)** | 0.92 | 0.96 | 0.94 | 23 |
| **Virginica (2)** | 0.95 | 0.90 | 0.93 | 21 |
| **Macro Average** | 0.96 | 0.95 | 0.95 | 60 |
| **Weighted Average** | 0.95 | 0.95 | 0.95 | 60 |

### Tuned Model
* **Final Test Accuracy:** `98.67%`

---

## 🛠️ Tech Stack & Requirements

* **Language:** Python 3.10+
* **Core Libraries:**
  * `numpy`
  * `pandas`
  * `scikit-learn`
  * `matplotlib`

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/iris-decision-tree-classifier.git](https://github.com/your-username/iris-decision-tree-classifier.git)
cd iris-decision-tree-classifier

### 2. Set Up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### 3. Install Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib notebook

### 4. Run the Notebook
```bash
jupyter notebook decision_tree_iris.ipynb

## 📂 Project Structure
├── decision_tree_iris.ipynb   # Main interactive lab notebook
└── README.md                  # Project documentation
