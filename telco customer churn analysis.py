# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 09:08:15 2026

@author: lenovo
"""

"""
customer churn analysis
churn prediction + marketing campaign analysis
"""


import pandas as pd
import numpy as np 
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, chi2_contingency, pearsonr, mannwhitneyu, norm


import statsmodels.api as sm
from statsmodels.formula.api import logit, ols
from statsmodels.stats.proportion import proportions_ztest 

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix 
import matplotlib.pyplot as plt


import seaborn as sns


import warnings 
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12,6)
plt.rcParams['font.size'] = 12

print("=" *80)
print("CUSTOMER ANALYTICS:  CHURN PREVENTION & MARKETING ROI")
print("=" *80)



#loading dataset
print("\n" + "="*80)
print("Step 1: Data loading & preparation")
print('=' * 80)


#churn dataset
print("\n1.1 Loading Telco Customer Churn Data..")
telco = pd.read_csv("C:/Users/lenovo/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv")

#telco.head()

telco['TotalCharges'] = pd.to_numeric(telco['TotalCharges'], errors = 'coerce')
telco = telco.dropna()
telco.isnull().sum()
telco["Churn"].value_counts(normalize = True)
telco['Churn'] = (telco['Churn'] == 'Yes').astype(int)

telco['SeniorCitizen'] = telco['SeniorCitizen'].astype(int)

#size of the data
#telco.shape[0]
#telco.shape[1]

#print(f"Telco data loaded: {telco.shape[0]} customers, {telco.shape[1]} feature")

#telco summary statistics
print(f"Overall Churn Rate: {telco['Churn'].mean() * 100:.1f}%")
print(f"Customer Tenure (months): Mean = {telco['tenure'].mean():.1f}, Median={telco['tenure'].median():.1f}")
print(f"Monthly Charges: ${telco['MonthlyCharges'].mean():.2f}")


print("\n" + "="*80)
print("STEP 3: HYPOTHESIS TESTING - CHURN ANALYSIS")
print("="*80)


#Contract Type Impact on Churn
print("\n" + "-"*60)
print("HYPOTHESIS 1: Contract Type and Churn Relationship")
print("-"*60)



contract_churn = pd.crosstab(telco['Contract'], telco['Churn'])
print("\nChurn by Contract Type:")
print(contract_churn)

chi2, p_contract, dof, expected = chi2_contingency(contract_churn)
print(f"\nChi-squared test: χ² = {chi2:.2f}, p-value = {p_contract:.10f}")

if p_contract < 0.001:
    print("STRONG EVIDENCE: Contract type siginificantly affects churn")
    monthly_churn_rate = telco[telco['Contract']== "Month-to_month"]['Churn'].mean()
    yearly_churn_rate = telco[telco['Contract'] == 'One year']['Churn'].mean()
    twoyear_churn_rate = telco[telco['Contract'] == 'Two year']['Churn'].mean()
    odds_monthly = monthly_churn_rate/ (1 - monthly_churn_rate)
    odds_yearly = yearly_churn_rate / (1- yearly_churn_rate)
    print(f"\nOdds Ratios:")
    print(f"  Month-to-month vs One year: {odds_monthly/ odds_yearly:.2f} x higher odds")
    print(f"  Months-to-month vs Two year: {odds_monthly/(twoyear_churn_rate/(1-twoyear_churn_rate)):.2f}x higher odds")

print("\n" + "-"* 60)
print("HYPOTHESIS 2: Tenure Comparison (Churned vs Retained)")
print("-"*60)


churned_tenure = telco[telco["Churn"] == 1]['tenure']
retained_tenure = telco[telco['Churn'] == 0]['tenure']

t_stat, p_tenure = ttest_ind(churned_tenure, retained_tenure, equal_var=False)
print(f"\nMean Tenure:")
print(f"  Churned: {churned_tenure.mean():.1f} months")
print(f"  Retained: {retained_tenure.mean():.1f} months")
print(f"  Difference: {retained_tenure.mean() - churned_tenure.mean():.1f} months")
print(f"\nT-test: t = {t_stat:.2f}, p-value = {p_tenure:.10f}")



if p_tenure < 0.001:
    print(" strong evidence: retained customers have significantly longer tenure")

    # calculating effect size (cohen's d)
    pooled_std = np.sqrt((churned_tenure.std() ** 2 + retained_tenure.std()**2)/2)
    cohens_d = (retained_tenure.mean() - churned_tenure.mean() / pooled_std)
    print(f"  Effect size (Cohen's d): {cohens_d:.2f} (Large effect)")





# 3.3 Monthly Charges Impact
print("\n" + "-"*60)
print("HYPOTHESIS 3: Monthly Charges and Churn Relationship")
print("-"*60)


churned_charges = telco[telco['Churn'] == 1]['MonthlyCharges']
retained_charges = telco[telco['Churn'] == 0]['MonthlyCharges']

t_stat_charge, p_charge = ttest_ind(churned_charges, retained_charges, equal_var = False)

print(f"\nMean Monthly Charges: ")
print(f"  Churned: ${churned_charges.mean():.2f}")
print(f"  Retained: ${retained_charges.mean():.2f}")
print(f"  Difference: ${churned_charges.mean() - retained_charges.mean():.2f}")
print(f"\nT-test: t = {t_stat_charge:.2f}, p-value = {p_charge:.10f}")


if p_charge < 0.001:
    print("Strong evidence: Churned customers have higher monthly charges")

print("\n" + "-"*60)
print("HYPOTHESIS 4: Payment Method Impact on Churn")
print("-"*60)



payment_churn = pd.crosstab(telco['PaymentMethod'], telco['Churn'])
print("\nChurn by payment mathod:")
print(payment_churn)





chi2_payment, p_payment, dof_payment, expected_payment = chi2_contingency(payment_churn)
print(f"\nChi-square test: χ² = {chi2_payment:.2f}, p-value = {p_payment:.6f}")

if p_payment < 0.001:
    print("✓ STRONG EVIDENCE: Payment method significantly affects churn")
    electronic_check_rate = telco[telco['PaymentMethod'] == 'Electronic check']['Churn'].mean()
    print(f"  Electronic check churn rate: {electronic_check_rate*100:.1f}% (Highest risk)")



# ============================================
# PART 5: LOGISTIC REGRESSION - CHURN PREDICTION
# ============================================

from sklearn.model_selection import train_test_split
print("\n" + "="*80)
print("STEP 5: PREDICTIVE MODELING - CHURN RISK")
print("="*80)

# Prepare features for churn model
features_churn = ['tenure', 'MonthlyCharges', 'SeniorCitizen']
categorical_features = ['Contract', 'PaymentMethod', 'InternetService']

# Create dummy variables
X_churn = pd.get_dummies(telco[features_churn + categorical_features], drop_first=True)
y_churn = telco['Churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_churn, y_churn, test_size=0.2, random_state=42)

# Train logistic regression
logit_model = LogisticRegression(max_iter=1000, random_state=42)
logit_model.fit(X_train, y_train)

# Predictions
y_pred = logit_model.predict(X_test)
y_pred_proba = logit_model.predict_proba(X_test)[:, 1]

# Model evaluation
print("\nLogistic Regression Results:")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))

# Feature importance (odds ratios)
feature_names = X_churn.columns
coefficients = logit_model.coef_[0]
odds_ratios = np.exp(coefficients)

print("\nTop 5 Churn Risk Factors (Odds Ratios):")
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Odds_Ratio': odds_ratios,
    'Coefficient': coefficients
}).sort_values('Odds_Ratio', ascending=False).head(5)

for idx, row in feature_importance.iterrows():
    print(f"  {row['Feature']}: OR = {row['Odds_Ratio']:.2f} "
          f"({(row['Odds_Ratio']-1)*100:+.1f}% change in odds)")


# ============================================
# PART 7: VISUALIZATIONS
# ============================================

print("\n" + "="*80)
print("STEP 7: GENERATING VISUALIZATIONS")
print("="*80)

# Create comprehensive visualizations
fig, axes = plt.subplots(3, 2, figsize=(16, 18))



# 1. Churn by Contract Type
contract_data = telco.groupby('Contract')['Churn'].mean() * 100
contract_data.plot(kind='bar', ax=axes[0,0], color='coral', edgecolor='black')
axes[0,0].set_title('Churn Rate by Contract Type', fontsize=14, fontweight='bold')
axes[0,0].set_ylabel('Churn Rate (%)')
axes[0,0].set_xlabel('Contract Type')
axes[0,0].axhline(y=telco['Churn'].mean()*100, color='red', linestyle='--', label='Overall Average')
axes[0,0].legend()

# 2. Tenure Distribution by Churn Status
for churn_status, color, label in [(0, 'green', 'Retained'), (1, 'red', 'Churned')]:
    axes[0,1].hist(telco[telco['Churn'] == churn_status]['tenure'], 
                   bins=30, alpha=0.6, color=color, label=label)
axes[0,1].set_title('Tenure Distribution by Churn Status', fontsize=14, fontweight='bold')
axes[0,1].set_xlabel('Tenure (months)')
axes[0,1].set_ylabel('Frequency')
axes[0,1].legend()


# 5. Feature Importance from Logistic Regression
feature_importance_sorted = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
}).sort_values('Coefficient', key=abs, ascending=False).head(8)

axes[2,0].barh(feature_importance_sorted['Feature'], feature_importance_sorted['Coefficient'])
axes[2,0].set_title('Top 8 Churn Predictors (Logistic Regression)', fontsize=14, fontweight='bold')
axes[2,0].set_xlabel('Coefficient')

# 6. Correlation Heatmap - Marketing Data

plt.tight_layout()
plt.savefig('churn_marketing_analysis.png', dpi=150, bbox_inches='tight')
print("churn_marketing_analysis.png")


























