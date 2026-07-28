# Streamlit link

https://ntifinalprojecthranalysis-2872026.streamlit.app/

# 👔 Employee Attrition Predictor

A machine learning project that predicts whether an employee is likely to leave a company (attrition), based on personal, job-related, and satisfaction attributes. The project includes model training/comparison in a Kaggle notebook and an interactive **Streamlit** web app for real-time predictions.

---

## 📌 Project Overview

Employee attrition is costly and disruptive for organizations — it leads to lost institutional knowledge, higher recruitment costs, and reduced productivity. This project builds and compares several machine learning and deep learning classification models to predict the likelihood of an employee leaving, so HR teams can identify at-risk employees and act proactively.

The project covers the full pipeline:
1. **Exploratory Data Analysis (EDA)** on the IBM HR Analytics dataset
2. **Preprocessing** — encoding categorical variables and feature scaling
3. **Model training & comparison** — Logistic Regression, Decision Tree, Random Forest, XGBoost, and a Neural Network
4. **Deployment** — an interactive multi-step Streamlit form that takes employee details and returns an attrition risk prediction

---

## 📊 Dataset Description

**Dataset:** [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

- **Records:** 1,470 employees
- **Features:** 35 columns, including:
  - **Demographics:** Age, Gender, MaritalStatus, DistanceFromHome
  - **Job info:** Department, JobRole, JobLevel, BusinessTravel, OverTime, EmployeeNumber
  - **Compensation:** MonthlyIncome, MonthlyRate, DailyRate, HourlyRate, PercentSalaryHike, StockOptionLevel
  - **Satisfaction & tenure:** JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager, TotalWorkingYears, NumCompaniesWorked, TrainingTimesLastYear
- **Target variable:** `Attrition` (`Yes` / `No`)
- **Dropped columns:** `Over18`, `EmployeeCount`, `StandardHours` (constant, single unique value — no predictive value)

---

## 🗂 Project Structure

```
employee-attrition-predictor/
│
├── notebooks/
│   └── grad-project-nti.ipynb     # EDA, preprocessing, model training & comparison (developed on Kaggle)
│
├── models/
│   ├── model.pkl                  # Trained classifier (best-performing model)
│   └── scaler.pkl                 # Fitted StandardScaler used during training
│
├── main.py                         # Streamlit web app (multi-step prediction form)
├── requirements.txt                # Python dependencies
└── README.md                      # Project documentation
```

> **Note:** Place your exported `model.pkl` and `scaler.pkl` inside a `models/` folder in the project root — `main.py` loads them from `models/model.pkl` and `models/scaler.pkl`.

---

## ⚙️ Installation Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/employee-attrition-predictor.git
   cd employee-attrition-predictor
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ How to Run the Project

### 1. Run the notebook (training & evaluation)
The notebook was developed and run on **Kaggle**. To reproduce the EDA, preprocessing, and model training/comparison steps:

1. Go to [Kaggle](https://www.kaggle.com/) and create a new notebook (or upload `grad-project-nti.ipynb` via **File → Upload Notebook**).
2. Add the dataset to the notebook: **Add Input** → search for `IBM HR Analytics Attrition Dataset` (`pavansubhasht/ibm-hr-analytics-attrition-dataset`) → **Add**.
3. Run all cells (**Run All**). The dataset path used in the notebook is:
   ```
   /kaggle/input/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv
   ```
4. After training, export the best model and scaler with `joblib` (e.g. `joblib.dump(model, "model.pkl")`, `joblib.dump(scaler, "scaler.pkl")`), then download them from the Kaggle output panel and place them in your local `models/` folder for the Streamlit app.

> Prefer running locally instead? Just open the notebook in Jupyter/VS Code and update the dataset path to point to your local copy of the CSV.

### 2. Run the Streamlit app (prediction UI)
Make sure `models/model.pkl` and `models/scaler.pkl` exist, then run:

```bash
streamlit run main.py
```

The app will open in your browser (default: `http://localhost:8501`). Fill in the employee's details across the 4-step form (Personal, Job Info, Compensation, Satisfaction & Tenure) and click **🚀 Predict Attrition Risk** to see the result.

---

## 🤖 Model Description

Five models were trained and compared on the preprocessed dataset (80/20 train-test split, stratified on the target):

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | 87% | Simple linear baseline |
| Decision Tree | 77% | Interpretable, but prone to overfitting |
| Random Forest | 83% | Ensemble of trees, reduces overfitting/variance |
| XGBoost | 87% | Gradient-boosted trees, strong performance |
| Neural Network (Shallow) | 91% | Dense network with early stopping |

**Why Decision Tree & Random Forest were included:**
- **Decision Tree** — chosen as an interpretable baseline; mirrors human decision-making logic and requires no feature scaling, making it easy to explain which features drive a prediction.
- **Random Forest** — chosen as an ensemble improvement over the single Decision Tree; it builds multiple trees on bootstrapped samples and aggregates their votes, reducing the overfitting and high variance that hurt the single-tree model (83% vs. 77% accuracy).

**Preprocessing applied to all models:**
- Label Encoding for categorical features (`BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `OverTime`)
- Standard scaling (`StandardScaler`) applied to all features before training
- Constant columns (`Over18`, `EmployeeCount`, `StandardHours`) removed during EDA

The best-performing model (based on accuracy and classification report metrics) is exported as `models/model.pkl` and served through the Streamlit app, alongside the fitted `models/scaler.pkl` used to scale new inputs identically to training data.

---

## 🌐 Streamlit Deployment Link

🔗 **[Live App](https://your-app-name.streamlit.app)**