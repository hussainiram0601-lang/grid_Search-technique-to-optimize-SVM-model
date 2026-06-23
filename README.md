# Hyperparameter Tuning using Grid Search & Kernel SVM

This project implements a **Kernel Support Vector Machine (SVM)** classifier optimized using **Grid Search** and validated via **k-Fold Cross Validation** on the `Social_Network_Ads.csv` dataset. It demonstrates how to systematically find the optimal hyperparameters (`C`, `gamma`, and choice of `kernel`) to maximize classification performance.

## 📌 Project Overview
When working with Support Vector Machines, picking default parameters rarely yields the best performance. This notebook walks through a robust machine learning workflow that transitions from data preparation to systematic model optimization, ensuring the final model generalizes perfectly to unseen consumer data.

---

## 🗺️ Machine Learning Pipeline Structure

The notebook is divided into clear, production-like steps:
1. **Data Preprocessing**: Importing core data utilities and parsing the `Social_Network_Ads.csv` dataset.
2. **Data Splitting**: Partitioning features into training ($75\%$) and test ($25\%$) sets.
3. **Feature Scaling**: Scaling features using standardization to prevent magnitude bias in distance metrics.
4. **Kernel SVM Training**: Fitting a non-linear kernel baseline model.
5. **k-Fold Cross Validation**: Evaluating baseline model resilience against variance and overfitting.
6. **Grid Search Optimization**: Auto-testing multiple combinations of hyperparameter parameters to select the highest-performing config.
7. **Decision Visualization**: Plotting 2D prediction regions for both training and test environments.

---

## 🛠️ Prerequisites & Tech Stack

Ensure you have the standard data science ecosystem installed:

```bash
pip install numpy pandas matplotlib scikit-learn
```

| Library | Primary Responsibility |
| :--- | :--- |
| **Python 3.x** | Operational Language |
| **Pandas** | Matrix parsing and data structure management |
| **NumPy** | Multidimensional arrays and vectorized computation |
| **Scikit-Learn** | Splitting, scaling, SVM modeling, Cross-Validation, and Grid Search tuning |
| **Matplotlib** | Training/Test decision boundary visualizations |

---

## 🚀 How to Run the Project

1. Clone this repository:
   ```bash
   git clone https://github.com
   ```
2. Place the `Social_Network_Ads.csv` file into the root project directory.
3. Open `grid_search.ipynb` inside **Google Colab** or **Jupyter Notebook**.
4. Run all cells sequentially (`Runtime > Run all` in Colab).

---

## 📈 Optimization Metrics
*(Tip: Update these placeholders with your final model outputs once the cells finish executing!)*

* **Baseline Accuracy (k-Fold)**: `XX.XX%`
* **Optimized Grid Search Parameters**: `{'C': X, 'gamma': X, 'kernel': 'rbf'}`
* **Final Test Set Accuracy**: `XX.XX%`

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
# grid_Search-technique-to-optimize-SVM-model
