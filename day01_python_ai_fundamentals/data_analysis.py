import pandas as pd
import numpy as np
import os
def analyze():
    #Here seed means to generate same random values every time we run the code
    np.random.seed(42)
    data={
        'customerid':range(101,201),
        'Age':np.random.randint(18,70,100),
        'Annualincome':np.random.randint(15,130,100),
        'SpendingScore':np.random.randint(1,100,100),
        'Purchasecategory':np.random.choice(['Electronics','Clothing','Groceries','Furniture'],100)
    }
    df=pd.DataFrame(data)
    #Here we use loc it is used to access a group of rows and columns by labels or a boolean array
   #Here we sue to_csv that means used to convert into csv file index is false emans pandas generally provides index as first column to rmeove that we use false
   #here panadas convert teh data into dfs and apply operations and then convert back df into data (like csv,excel,html)


    df.loc[5:10,'Annualincome']=np.nan
    csv_path='day01_python_ai_fundamentals/customer_data.csv'
    df.to_csv(csv_path,index=False)
    print(f"\nDtaset{csv_path}")
    read=pd.read_csv(csv_path)#Used to load csv
    print(read.head())#Means first 5 rows
    print("\n Data info :It provides all the datatypes and if anything is missing")
    print(read.info())
    
    print("\n Data cleaning handling missing values")
    missing=read['Annualincome'].isnull().sum()
    print(f"missing value:{missing}")
    income_median=read['Annualincome'].median()
    read['Annualincome']=read['Annualincome'].fillna(income_median)
    print(f"median value of annual incpme{income_median}")
    # .describe() gives us the count, mean, std, min, max instantly
    stats=read.describe()
    print(stats)
    # Correlation: Does Age affect Spending Score?
    # A number close to 1 means yes. Close to 0 means no connection.
    correlation=read['Age'].corr(read['SpendingScore'])
    print(f"Corelalatoion value is {correlation:.4f}")
    print("\n--- 5. Grouping Data ---")
    # Who spends the most? Grouping by Category
    category_spend = read.groupby('Purchasecategory')['SpendingScore'].mean()
    print("Average Spending Score by Category:")
    print(category_spend)

if __name__ == "__main__":
    analyze()