import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(df=None, df_path='data/employee_salary_data.csv', output_dir='static/plots'):
    os.makedirs(output_dir, exist_ok=True)
    

    if df is None:
        df = pd.read_csv(df_path)
    else:
        df = df.copy()
    

    sns.set_palette("viridis")
    plt.style.use('bmh')
    

    print("Generating feature histograms...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = 4
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f'Distribution of {col}', fontsize=12)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(f'{output_dir}/histograms.png', dpi=300)
    plt.close()

    print("Generating correlation heatmap...")
    plt.figure(figsize=(16, 10))
    # Select only numeric for correlation
    corr = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, square=True)
    plt.title('Correlation Heatmap of Employee Features', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap.png', dpi=300)
    plt.close()

    print("Generating scatter plots vs salary...")
    top_features = ['experience', 'performance', 'projects', 'leadership_score', 'skill_score']

    top_features = [col for col in top_features if col in df.columns]
    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(top_features):
        sns.scatterplot(data=df, x=col, y='salary', alpha=0.5, ax=axes[i], color='purple')
        sns.regplot(data=df, x=col, y='salary', scatter=False, ax=axes[i], color='red')
        axes[i].set_title(f'Salary vs {col}', fontsize=14)
        
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatter_salary.png', dpi=300)
    plt.close()

    print("Generating categorical boxplots...")
    cat_cols = ['education', 'job_level', 'department']

    cat_cols = [col for col in cat_cols if col in df.columns]
    
    if cat_cols:
        n_cats = len(cat_cols)
        fig, axes = plt.subplots(1, n_cats, figsize=(6 * n_cats, 6))
        if n_cats == 1:
            axes = [axes]
        for i, col in enumerate(cat_cols):
            sns.boxplot(data=df, x=col, y='salary', ax=axes[i], palette='Set2')
            axes[i].set_title(f'Salary by {col}', fontsize=14)
            axes[i].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.savefig(f'{output_dir}/boxplots.png', dpi=300)
        plt.close()
    
    print(f"EDA plots saved to {output_dir}")

if __name__ == "__main__":
    if not os.path.exists('data/employee_salary_data.csv'):
        from data_generator import generate_employee_data
        df = generate_employee_data()
    else:
        df = pd.read_csv('data/employee_salary_data.csv')
    run_eda()
