import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def cosmetics_dataset_analysis(file_path):
    """
    Perform comprehensive Exploratory Data Analysis (EDA) on the cosmetics dataset.
    
    Returns:
    --------
    A dictionary containing key analysis results
    """
    df = pd.read_csv(file_path)
    
    # 1. Dataset Size Analysis
    print("1. Dataset Size Analysis")
    print(f"Total number of rows: {len(df)}")
    print(f"Total number of columns: {len(df.columns)}")
    print("\n")
    
    # 2. Missing Values Analysis
    print("2. Missing Values Analysis")
    missing_values = df.isnull().sum()
    print("Missing values per column:")
    print(missing_values[missing_values > 0] if any(missing_values > 0) else "No missing values")
    print("\n")
    
    # 3. Duplicate Rows Analysis
    print("3. Duplicate Rows Analysis")
    duplicate_rows = df[df.duplicated()]
    print(f"Number of duplicate rows: {len(duplicate_rows)}")
    print("\n")
    
    # 4. Data Type Consistency
    print("4. Data Type Consistency")
    print(df.dtypes)
    print("\n")
    
    # 5. Price Range Analysis
    print("5. Price Range Analysis")
    price_stats = df['Price'].describe()
    print("Price Statistics:")
    print(price_stats)
    
    # Identify potential price outliers using IQR method
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    price_outliers = df[(df['Price'] < lower_bound) | (df['Price'] > upper_bound)]
    print("\nPrice Outliers:")
    print(price_outliers[['Brand', 'Name', 'Price']])
    print(f"\nNumber of price outliers: {len(price_outliers)}")
    print("\n")
    
    # 6. Rank Range Analysis
    print("6. Rank Range Analysis")
    rank_stats = df['Rank'].describe()
    print("Rank Statistics:")
    print(rank_stats)
    print("\n")
    
    # 7. Skin Type Columns Analysis
    print("7. Skin Type Columns Analysis")
    skin_type_columns = ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']
    
    print("Unique values in skin type columns:")
    for col in skin_type_columns:
        print(f"{col}: {df[col].unique()}")
    
    print("\nSkin Type Distribution:")
    skin_type_dist = df[skin_type_columns].sum()
    print(skin_type_dist)
    
    plt.figure(figsize=(10, 6))
    skin_type_dist.plot(kind='bar')
    plt.title('Skin Type Distribution')
    plt.xlabel('Skin Type')
    plt.ylabel('Number of Products')
    plt.tight_layout()
    plt.savefig('skin_type_distribution.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    df['Price'].hist(bins=30)
    plt.title('Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('price_distribution.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    df['Rank'].hist(bins=6)
    plt.title('Rank Distribution')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('rank_distribution.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Price'], df['Rank'], alpha=0.5)
    plt.title('Price vs Rank')
    plt.xlabel('Price')
    plt.ylabel('Rank')
    plt.tight_layout()
    plt.savefig('price_vs_rank.png')
    plt.close()
    
    return {
        'total_rows': len(df),
        'missing_values': missing_values.to_dict(),
        'duplicate_rows': len(duplicate_rows),
        'price_stats': price_stats.to_dict(),
        'rank_stats': rank_stats.to_dict(),
        'skin_type_distribution': skin_type_dist.to_dict()
    }

if __name__ == "__main__":
    file_path = 'cosmetics.csv' 
    analysis_results = cosmetics_dataset_analysis(file_path)
  
    print("\nAnalysis Summary:")
    for key, value in analysis_results.items():
        print(f"{key}: {value}")