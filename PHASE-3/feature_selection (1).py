import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import os

def feature_selection_visualization(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'salary']
    
    X = df[numeric_cols]
    y = df['salary']

    X = X.fillna(X.mean())
    
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_scores = pd.Series(mi_scores, index=numeric_cols).sort_values(ascending=False)

    os.makedirs('static/plots', exist_ok=True)
    plt.figure(figsize=(10, 6))
    mi_scores.plot(kind='barh', color='steelblue')
    plt.xlabel('Mutual Information Score')
    plt.title('Feature Importance for Salary Prediction')
    plt.tight_layout()
    plt.savefig('static/plots/feature_importance.png', dpi=300)
    plt.close()
    
    print("Feature Importance plot saved to static/plots/feature_importance.png")
    return list(mi_scores.head(10).index)  # Return top 10 features
