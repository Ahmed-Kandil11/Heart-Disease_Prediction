# ❤️ Heart Disease Prediction Project

This project applies machine learning to predict the likelihood of heart disease using the UCI Heart Disease dataset.
It demonstrates the full lifecycle of a data science project:

Data cleaning and preprocessing

Exploratory Data Analysis (EDA)

Feature selection and dimensionality reduction

Training and evaluating multiple ML models

Hyperparameter tuning for model stability

Deployment as an interactive Streamlit web app

🚀 The final deployed app allows users to enter patient health data and instantly receive a prediction, along with the probability of having heart disease.

---

## 📊 Project Workflow

1. **Data Preprocessing**
   - Handled missing values (median/mode imputation).
   - Encoded categorical variables with `OneHotEncoder`.
   - Scaled numerical features using `StandardScaler`.
   - Performed Exploratory Data Analysis (EDA).

2. **Feature Selection**
   - Random Forest Feature Importance.
   - Recursive Feature Elimination (RFE).
   - Chi-Square Test.
   - Final reduced dataset saved for modeling.

3. **Modeling**
   - Models trained: Logistic Regression, Decision Tree, Random Forest, SVM.
   - Evaluated with Accuracy, Precision, Recall, F1, and AUC.
   - **Random Forest** selected as the best-performing model.

4. **Hyperparameter Tuning**
   - GridSearchCV & RandomizedSearchCV applied.
   - Verified model stability (no major improvement → default Random Forest chosen).

5. **Deployment**
   - Final model stored as `heart_disease_pipeline.pkl`.
   - Streamlit app created for real-time predictions.
   - Hosted on **Streamlit Community Cloud**.

---

## 🖥️ Usage

### Run locally
```bash
streamlit run app.py

.
├── App.py                        # Streamlit web app for real-time predictions
├── heart_disease_pipeline.pkl    # Trained Random Forest model with preprocessing pipeline
├── requirements.txt              # List of required Python libraries
├── notebooks/ # Development notebooks (order shows workflow)
│ ├── preprocessing1.ipynb           # Notebook 1: data cleaning, encoding, scaling, and EDA
│ ├── notebook2_PCA.ipynb # Notebook 2: (Experimental) PCA analysis and explained variance plots (dimensionality reduction)
│ ├── notebook3_feature_selection.ipynb       # Notebook 3:(Experimental) feature importance, RFE, Chi-Square, final reduced dataset
│ ├── notebook4_Supervised-learning_classifiaction.ipynb     # Notebook 4: model training, evaluation, hyperparameter tuning
│ ├── notebook6_full_pipeline.ipynb                # Final combined notebook (end-to-end workflow + exportable model)
> 📝 Note: Notebooks **2_pca.ipynb** and **3_feature_selection.ipynb** are included for transparency and experimentation only.  
> They showcase intermediate analysis (PCA and feature selection) but are **not part of the final deployed pipeline**, which is fully contained in **pipeline.ipynb**.

