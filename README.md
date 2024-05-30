# Analysis of Cardiotocography Features and Their Impact on Foetal Health

## Project Overview

Eradicating preventable child and maternal deaths by 2030 is a crucial target of the UN Sustainable Development Goals. In low-income settings, accessible and effective prenatal care remains a challenge. This project leverages the potential of cardiotocograms (CTGs), an inexpensive and widely available tool, to analyze foetal health data and predict potential complications. By unraveling the interconnected relationships between various CTG features, we aim to develop a classification system that empowers early intervention and ultimately saves lives. This data-driven approach holds immense promise for improving maternal and child health outcomes, particularly in resource-constrained environments.

## Table of Contents
1. [Importing Data Dependencies](#1-importing-data-dependencies)
2. [Loading Data](#2-loading-data)
3. [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
4. [Preprocessing](#4-preprocessing)
5. [Model and Model Evaluation](#5-model-and-model-evaluation)
6. [Conclusion and Recommendations](#6-conclusion-and-recommendations)

## 1. Importing Data Dependencies
In this section, we import all necessary libraries and dependencies required for the project. These include libraries for data manipulation, visualization, and machine learning.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
```

## 2. Loading Data
Here, we load the cardiotocography dataset, which contains various features related to foetal health, from a CSV file into a Pandas DataFrame.

```python
data = pd.read_csv('path_to_ctg_data.csv')
```

## 3. Exploratory Data Analysis (EDA)
In the EDA section, we perform a comprehensive analysis of the dataset to uncover patterns, relationships, and anomalies. This involves visualizing the distribution of features, examining correlations, and identifying any missing values.

```python
# Example of a correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()
```

## 4. Preprocessing
This section involves cleaning the data, handling missing values, and standardizing numerical features to prepare the data for modeling.



## 5. Model and Model Evaluation
We build and evaluate machine learning models to classify foetal health status based on CTG features. Model performance is assessed using metrics such as accuracy, precision, recall, and F1-score.

```python
# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(scaled_data, data['target'], test_size=0.2, random_state=42)

# Training a Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions and evaluation
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
```

## 6. Conclusion and Recommendations
In the final section, we summarize the findings of the project and provide recommendations for future work. This includes potential improvements to the model, additional features that could be explored, and the importance of this work in improving maternal and child health outcomes.

## Contributing
Contributions to this project are welcome. Please feel free to fork this repository and submit pull requests.

## License
This project is licensed under the MIT License.

---

By exploring the features of CTGs and developing predictive models, this project aims to make a meaningful impact on prenatal care in resource-constrained settings, ultimately contributing to the global goal of eradicating preventable child and maternal deaths.
