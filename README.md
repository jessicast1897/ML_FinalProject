# ML_FinalProject
# Gun Violence Early Warning System Using XGBoost

## Project Overview

This project builds a machine learning early warning system to predict whether a U.S. county has a **high** or **low** level of gun violence based on socioeconomic, demographic, and social vulnerability indicators.

The final individual model uses **XGBoost**, a gradient-boosted decision tree algorithm that performs well on structured tabular data. The purpose of this project is to support public agencies, policymakers, school districts, healthcare organizations, and community groups in identifying counties that may need targeted prevention resources before violence escalates.

This project is intended for public health and prevention planning, not punitive enforcement.

---

## Predictive Question

### Primary Predictive Question

Given socioeconomic, demographic, and social vulnerability features for a U.S. county, can we accurately predict whether that county has a **high level** of gun violence?

### Secondary Predictive Question

What are the top features that have the most influence on whether a county is classified as high gun violence?

---

## Data Sources

The project combines county-level data from multiple public sources, including:

- CDC firearm mortality data
- Social Vulnerability Index, also known as SVI, features
- ACS / Census socioeconomic and demographic indicators

The dataset includes information such as:

- County
- State
- Year
- Deaths
- Population
- Firearm death crude rate
- Median household income
- Poverty indicators
- Unemployment rate
- Housing burden
- Disability rate
- Minority population indicators
- Age 65+ indicators
- Multi-unit housing indicators
- Other social vulnerability features

---

## Dataset Summary

The final modeling dataset contains:

- **13,328 rows**
- **32 original columns**
- County-year observations
- Years from **2019 through 2023**
- No missing values
- No duplicate records

The target column is:

```text
violence_level
```

The target variable classifies each county-year observation as either:

```text
high
low
```

---

## Target Variable Creation

The target variable was created using the median firearm crude rate.

Counties with a crude rate greater than or equal to the median were labeled as:

```text
high
```

Counties with a crude rate below the median were labeled as:

```text
low
```

The median crude rate threshold used in the project was approximately:

```text
20.64 firearm deaths per 100,000 people
```

This turns the project into a binary classification problem.

---

## Data Leakage Prevention

An important part of the project was identifying and removing data leakage.

Some variables were removed because they directly or indirectly reveal the answer the model is supposed to predict.

The following columns were dropped before final modeling:

```text
violence_level
county
state
year
crude_rate
FIPS
deaths
population
yoy_pct_change
```

These columns were removed because:

```text
crude_rate = deaths / population * 100,000
```

Since the target variable was created from crude rate, keeping `crude_rate`, `deaths`, or `population` would allow the model to learn the answer directly instead of learning from socioeconomic and demographic patterns.

After removing leakage variables, the model performance became more realistic and more appropriate for deployment.

---

## Feature Selection

After removing leakage variables, the final model used legitimate socioeconomic, demographic, and social vulnerability predictors.

The final feature set included SVI and ACS-style indicators such as:

- Median household income
- Unemployment rate
- Poverty-related variables
- Housing burden
- Multi-unit housing
- Disability rate
- Age 65+ population
- Minority status
- Limited English indicators
- Other county-level vulnerability indicators

The final dataset used approximately **26 modeling columns** after feature selection and leakage removal.

---

## Candidate Models

Several machine learning models were considered for this classification task:

### Logistic Regression

Logistic Regression was considered because it is interpretable and useful for binary classification problems.

### Random Forest

Random Forest was considered because it handles many predictor variables, captures nonlinear relationships, and performs well on structured data.

### XGBoost

XGBoost was selected for this individual model because it works well with tabular data, handles nonlinear feature relationships, supports regularization, and often performs strongly on classification problems.

### Neural Network

A Neural Network was considered because it can capture complex feature interactions, although it may require more tuning and interpretability work.

---

## Final Model

The final individual model uses:

```text
XGBoost Classifier
```

XGBoost was chosen because it:

- Performs well on structured tabular datasets
- Captures nonlinear patterns
- Handles complex feature interactions
- Includes regularization to reduce overfitting
- Works well with feature importance and SHAP interpretability
- Does not require strict linear assumptions

---

## Model Training Workflow

The notebook follows this general workflow:

1. Import required Python libraries
2. Load the dataset
3. Review data quality
4. Check for missing values and duplicates
5. Explore descriptive statistics
6. Create the `violence_level` target variable
7. Identify and remove data leakage variables
8. Separate features and target variable
9. Encode categorical variables if needed
10. Split the data into training and testing sets
11. Train an initial XGBoost model
12. Evaluate baseline performance
13. Tune hyperparameters to reduce overfitting
14. Train the final leakage-free XGBoost model
15. Evaluate final model performance
16. Generate confusion matrix and ROC curve
17. Review feature importance
18. Use SHAP analysis for explainability
19. Save the trained model and feature list

---

## Train/Test Split

The data was split into training and testing sets using a stratified split.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

A stratified split was used so that the proportion of `high` and `low` violence labels stayed balanced in both the training and testing datasets.

---

## Hyperparameter Tuning

The original model showed signs of overfitting, especially when leakage variables were included.

To improve generalization, the XGBoost model was tuned using regularization and more conservative tree settings.

The final tuned model used parameters similar to:

```python
XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.5,
    min_child_weight=5,
    random_state=42,
    eval_metric="logloss"
)
```

### Tuning Goals

The tuning process focused on:

- Reducing overfitting
- Improving test performance
- Improving generalization
- Maintaining strong recall for high-risk counties
- Creating a realistic leakage-free model

---

## Performance Metrics

The model was evaluated using several classification metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- ROC Curve
- Train/Test accuracy gap

These metrics were selected because the project is a binary classification problem.

Recall is especially important because a false negative means the model predicts a county as low risk when it is actually high risk.

---

## Final Model Results

The final tuned leakage-free XGBoost model achieved approximately:

```text
Train Accuracy: 0.8856
Test Accuracy: 0.8492
Overfit Gap: 0.0364
F1 Score: 0.8495
ROC-AUC: 0.9306
```

The final model achieved strong performance while avoiding direct leakage from the target variable.

The earlier model had higher performance, but that performance was inflated because leakage variables were still included. After removing those variables, the results became more realistic and trustworthy.

---

## Top Features Predicting High Gun Violence

The final XGBoost model identified the following as the most important features:

```text
1. svi_multi_unit        importance = 0.1931
2. acs_median_income    importance = 0.0774
3. svi_disabled         importance = 0.0457
4. svi_housing_burden   importance = 0.0453
5. spike                importance = 0.0448
6. svi_age65_plus       importance = 0.0424
7. svi_unemp_rate       importance = 0.0416
8. acs_unemp_rate       importance = 0.0414
```

These results suggest that socioeconomic vulnerability, housing conditions, unemployment, disability status, and age-related vulnerability are important signals in predicting county-level gun violence risk.

---

## Model Interpretability

The project uses model interpretability methods to understand how the model makes predictions.

### Feature Importance

XGBoost feature importance was used to identify which variables contributed most to the model's predictions.

### SHAP Analysis

SHAP was used to explain how individual features influenced predictions.

SHAP helps show whether a feature pushes a prediction toward `high` or `low` gun violence risk.

---

## Key Findings

The final model suggests that county-level gun violence risk is strongly associated with social and economic vulnerability.

Important signals included:

- Multi-unit housing
- Lower median household income
- Housing cost burden
- Disability rate
- Older population indicators
- Unemployment indicators

The top features suggest that urban socioeconomic deprivation is a major signal in predicting high gun violence risk.

---

## Recommendations

Based on the model results, prevention efforts could focus on counties with higher socioeconomic vulnerability.

Recommended interventions include:

### Education and Career Support

- Support school attendance programs
- Invest in classroom resources
- Provide career counseling in high school
- Expand higher education pathways
- Increase job training opportunities

### Housing and Health Stability

- Support affordable housing
- Expand eviction prevention programs
- Invest in neighborhood stabilization
- Improve access to trauma care
- Increase insurance enrollment support
- Expand access to mental health resources

### Community Prevention

- Use predictions to guide supportive resources
- Prioritize early intervention
- Support violence prevention programs
- Avoid punitive or stigmatizing use of model results

---

## Files in This Repository

This repository may include:

```text
Machine_Learning_Final_Model_xgboost.ipynb
```

Main Jupyter Notebook containing the XGBoost modeling workflow.

```text
gun_violence_with_levels.csv
```

Dataset used for model training and evaluation.

```text
xgboost_machine_learning_final_model.sav
```

Saved trained XGBoost model.

```text
model_features.pkl
```

Saved list of model features used during training.

---

## Requirements

The project uses Python and the following libraries:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
pickle
joblib
```

Install the required packages with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap joblib
```

---

## How to Run the Project

1. Clone this repository.

```bash
git clone https://github.com/your-username/your-repository-name.git
```

2. Navigate into the project folder.

```bash
cd your-repository-name
```

3. Install the required packages.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap joblib
```

4. Make sure the dataset is in the project folder.

```text
gun_violence_with_levels.csv
```

5. Open the notebook.

```text
Machine_Learning_Final_Model_xgboost.ipynb
```

6. Run all cells in the notebook.

7. After the notebook finishes running, the final model and feature list should be saved.

```text
xgboost_machine_learning_final_model.sav
model_features.pkl
```

---

## Loading the Saved Model

The trained model can be loaded using Python:

```python
import pickle

with open("xgboost_machine_learning_final_model.sav", "rb") as file:
    model = pickle.load(file)

with open("model_features.pkl", "rb") as file:
    model_features = pickle.load(file)
```

---

## Possible Streamlit Deployment

This model can be used in a Streamlit app where a user enters county-level socioeconomic and demographic values, and the app predicts whether the county is at high or low risk for gun violence.

A deployed Streamlit demo was also created for this project:

```text
https://gun-violence-early-warning-system.streamlit.app/
```

---

## Limitations

This project has several limitations:

- The target variable is based on a median split, which simplifies gun violence into only two categories.
- A regression model could preserve more detail by predicting the actual crude rate.
- A multi-class model could classify counties as low, medium, or high risk.
- The model uses a stratified random split rather than a temporal split.
- A temporal split, such as training on 2019–2021 and testing on 2022–2023, would better simulate real-world deployment.
- County-level predictions can hide neighborhood-level variation within counties.
- Specific legislation, executive orders, and local policy changes were not directly measured.
- The model is based on historical data and may need retraining as patterns change.

---

## Future Improvements

Future versions of this project could include:

- Temporal train/test split for more realistic deployment testing
- Multi-class outcome such as low, medium, and high risk
- Regression model to predict firearm death crude rate directly
- Additional features such as gun law index, opioid mortality rate, healthcare access, and social cohesion measures
- More local-level data, such as census tract or ZIP code data
- Model monitoring for future deployment
- Improved Streamlit interface
- Comparison with LightGBM, CatBoost, Random Forest, Logistic Regression, and Neural Network models

---

## Ethical Considerations

This project should be used only for prevention, public health planning, and supportive resource allocation.

The model should not be used to stigmatize counties, communities, or demographic groups.

Predictions should support interventions such as:

- Mental health resources
- Community violence prevention
- Housing support
- Education programs
- Job training
- Trauma care access
- Public health funding

The goal is to identify areas where support may be needed, not to punish or label communities.

---

## Conclusion

The final leakage-free XGBoost model was able to predict high versus low county-level gun violence with strong performance.

The model achieved approximately:

```text
Test Accuracy: 84.92%
F1 Score: 84.95%
ROC-AUC: 93.06%
```

The results show that socioeconomic, demographic, and social vulnerability features can be useful in identifying counties at higher risk for gun violence.

The most important predictors suggest that urban socioeconomic deprivation, housing instability, disability, unemployment, and income-related factors are major signals in the model.

This project demonstrates how machine learning can be used as an early warning tool to support proactive and prevention-focused public health decision-making.

---

## Author

Jessica, Josh, Jake, Sebastian

BSAN 6070  
Spring 2026  
Individual Model: XGBoost
