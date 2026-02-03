import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def visualize():
    df = pd.read_csv('day01_python_ai_fundamentals/customer_data.csv')
    output_dir = 'day01_python_ai_fundamentals/outputs'
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print("--- Generating Visualizations ---")

    # CHART 1: Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
    plt.title('Distribution of Customer Ages')
    plt.xlabel('Age')
    plt.ylabel('Number of Customers')
    
    # --- FIX IS HERE: Added filename '/1_age_distribution.png' ---
    plt.savefig(f'{output_dir}/1_age_distribution.png') 
    print("1. Saved Age Histogram")
    plt.close()

    # CHART 2: Scatter Plot
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x='Annualincome', y='SpendingScore', hue='Purchasecategory')
    plt.title('Income vs Spending Score')
    plt.savefig(f'{output_dir}/2_income_vs_spending.png')
    print("2. Saved Scatter Plot")
    plt.close()

    # CHART 3: Bar Chart
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x='Purchasecategory', y='SpendingScore')
    plt.title('Average Spending Score by Category')
    plt.savefig(f'{output_dir}/3_category_bar_chart.png')
    print("3. Saved Bar Chart")
    plt.close()

    # CHART 4: Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['Annualincome'], color='lightgreen')
    plt.title('Box Plot of Annual Income')
    plt.savefig(f'{output_dir}/4_income_boxplot.png')
    print("4. Saved Box Plot")
    plt.close()

    # CHART 5: Heatmap
    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig(f'{output_dir}/5_correlation_heatmap.png')
    print("5. Saved Heatmap")
    plt.close()

    print(f"\nAll plots saved to {output_dir}")

if __name__ == "__main__":
    visualize()