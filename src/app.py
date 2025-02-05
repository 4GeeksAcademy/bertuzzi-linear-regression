from utils import db_connect
engine = db_connect()

# your code here
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

 #%%

insurance_data = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv')
print(insurance_data.head())
print(insurance_data.describe())
print(insurance_data.info())

# Drop duplicated values
print(f'Duplicate values: {insurance_data.duplicated().sum()}')
if insurance_data.duplicated().sum() != 0:
    insurance_data.drop_duplicates(inplace=True)

# Convert categorical values to numeric
cat_dimensions = ['sex', 'smoker', 'region']
# Dictionary to store mapping
factorize_mappings = {}

for col in cat_dimensions:
    encoded_value, categories = pd.factorize(insurance_data[col])
    factorize_mappings[col] = dict(enumerate(categories))
    insurance_data[col] = encoded_value
print(insurance_data.head())
print(insurance_data.info())
print(factorize_mappings)

# Univariate data analysis

fig, axes = plt.subplots(4,3, figsize=(15,15))

sns.histplot(data=insurance_data, x='age', kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Age Distribution')
sns.histplot(data=insurance_data, x='bmi', kde=True, ax=axes[0, 1])
axes[0, 1].set_title('BMI Distribution')
sns.histplot(data=insurance_data, x='charges', kde=True, ax=axes[0, 2])
axes[0, 2].set_title('Charges Distribution')
sns.boxplot(data=insurance_data, x='age', ax=axes[1, 0])
axes[1, 0].set_title('Age Box Plot')
sns.boxplot(data=insurance_data, x='bmi', ax=axes[1, 1])
axes[1, 1].set_title('BMI Box Plot')
sns.boxplot(data=insurance_data, x='charges', ax=axes[1, 2])
axes[1, 2].set_title('Charges Box Plot')
sns.countplot(data=insurance_data, x='children', ax=axes[2, 0])
axes[2, 0].set_title('Children Count')
sns.countplot(data=insurance_data, x='sex', ax=axes[2, 1])
axes[2, 1].set_title('Sex Count')
sns.countplot(data=insurance_data, x='smoker', ax=axes[2, 2])
axes[2, 2].set_title('Smoker Count')
sns.countplot(data=insurance_data, x='region', ax=axes[3, 0])
axes[3, 0].set_title('Region Count')
axes[3, 1].set_visible(False)
axes[3, 2].set_visible(False)

# Multivariate data analysis

plt.figure(figsize=(15,15))
sns.pairplot(insurance_data)

numeric_features = ['age', 'bmi']

fig, axes = plt.subplots(1, len(numeric_features), figsize=(15, 5))
for i, feature in enumerate(numeric_features):
    sns.regplot(x=insurance_data[feature], y=insurance_data['charges'], ax=axes[i], scatter_kws={'alpha': 0.5})
    axes[i].set_title(f'Regression: {feature} vs. Charges')

plt.figure(figsize=(8, 6))
sns.heatmap(insurance_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.show()

# Handle outliers
print(len(insurance_data[insurance_data['charges'] >= 50000]))
high_carges = insurance_data[insurance_data['charges'] >= 50000]
print(high_carges)
corr1_charges_bmi = np.corrcoef(insurance_data['charges'], insurance_data['bmi'])
print(f'Overall correlation level: {corr1_charges_bmi}')
corr2_charges_bmi = np.corrcoef(high_carges['charges'], high_carges['bmi'])
print(f'High tail correlation level: {corr2_charges_bmi}')
# It seems that there is a relationship between BMI and high charges, probably higher premiums due to high weight. Do not drop outliers.

# Split train and test data
from sklearn.model_selection import train_test_split

X = insurance_data[['age', 'bmi', 'children', 'smoker', 'sex']]
y = insurance_data['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=117)

# Normalize dataset
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train_norm = scaler.transform(X_train)
X_train_norm = pd.DataFrame(X_train_norm, index=X_train.index, columns=X_train.columns)
X_test_norm = scaler.transform(X_test)
X_test_norm = pd.DataFrame(X_test_norm, index=X_test.index, columns=X_test.columns)

# Perform linear regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train_norm, y_train)

a = print(f'Model intercept: {model.intercept_}')
coeffs = print(f'Model coefficients: {model.coef_}')

predictions = model.predict(X_test_norm)

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f'Model mean squared error: {mse}')
print(f'Model r2 score: {r2}')