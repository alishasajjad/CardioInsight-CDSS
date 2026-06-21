# Clinical Features in Heart Disease ML Models (Project Documentation)

## Cleveland / Statlog Feature Definitions

| Feature | Description |
|---------|-------------|
| Age | Age in years |
| Sex | 1 = male, 0 = female |
| Chest pain type | 0 typical angina, 1 atypical, 2 non-anginal, 3 asymptomatic |
| Resting BP | Systolic blood pressure at rest (mmHg) |
| Cholesterol | Serum cholesterol (mg/dl) |
| Fasting blood sugar | 1 if > 120 mg/dl |
| Resting ECG | 0 normal, 1 LV hypertrophy, 2 ST-T abnormality |
| Max heart rate | Maximum heart rate achieved |
| Exercise angina | 1 if exercise-induced angina |
| Oldpeak | ST depression induced by exercise |
| ST slope | 0 upsloping, 1 flat, 2 downsloping |
| Major vessels | Number of major vessels colored by fluoroscopy (0-3) |
| Thalassemia | 0 normal, 1 fixed defect, 2 reversible defect |

## Interpretation Notes
- **Oldpeak** and **ST slope** relate to ischemia on stress testing.
- **Major vessels** and **thalassemia** reflect angiographic findings.
- **Asymptomatic chest pain type** is common in severe disease presentations in some cohorts.

## Ensemble System
This project combines Random Forest, XGBoost, and ANN predictions using ROC-AUC–weighted averaging for a final risk score.

## Educational Use
Model outputs support learning and decision-support demonstrations; they are not FDA-cleared diagnostic devices.
