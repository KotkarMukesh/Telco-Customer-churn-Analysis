# Telco-Customer-churn-Analysis
This project analyzes Telco customer data to identify churn patterns, test key hypotheses, and build a predictive model to help prevent customer churn and improve marketing ROI.




### Dataset Overview
- **Overall Churn Rate:** 26.6%
- **Average Customer Tenure:** 32.4 months
- **Average Monthly Charges:** $64.80


Hypothesis Testing Results

###  Contract Type & Churn
- Chi-squared test: χ² = 1179.55, p-value ≈ 0.0
- Strong Evidence: Contract type significantly affects churn
- Month-to-month customers churn the most



### Tenure Comparison (Churned vs Retained)
- Churned customers avg tenure: **18.0 months**
- Retained customers avg tenure: **37.7 months**
- Difference: **19.7 months**
- ✅ Strong Evidence: Retained customers stay significantly longer



### Monthly Charges & Churn
- Churned customers pay: **$74.44/month**
- Retained customers pay: **$61.31/month**
- ✅ Strong Evidence: Higher charges linked to more churn




### Payment Method & Churn
- Electronic check has the highest churn rate: **45.3%**
- ✅ Strong Evidence: Payment method significantly affects churn


## Predictive Model (Logistic Regression)
| Metric | Score |
|--------|-------|
| ROC-AUC | 0.825 |
| Accuracy | 79% |
| Precision (Churned) | 63% |
| Recall (Churned) | 49% |

### Top 5 Churn Risk Factors
| Factor | Odds Ratio | Impact |
|--------|-----------|--------|
| Fiber Optic Internet | 2.44 | +143.6% |
| Senior Citizen | 1.49 | +49.4% |
| Electronic Check Payment | 1.47 | +47.4% |
| Monthly Charges | 1.01 | +0.6% |
| Tenure | 0.97 | -3.2% |

## Tools Used
- Python
- Spyder IDE
- Pandas
- Scikit-learn
- Matplotlib/Seaborn

---

## Author
Mukesh Kotkar
