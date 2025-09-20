# ❤️ Heart Disease Prediction Project

This project predicts the likelihood of heart disease using the **UCI Heart Disease dataset**.  
It includes end-to-end steps: preprocessing, feature selection, model training, hyperparameter tuning, and deployment with **Streamlit Cloud**.

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
