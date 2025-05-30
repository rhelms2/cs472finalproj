import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Brought to you by ChatGPT

df = pd.read_csv('Final Project/calories.csv')

# List of numeric columns except Calories
numeric_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

plt.figure(figsize=(15, 10))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 3, i)
    sns.regplot(data=df, x=col, y='Calories', scatter_kws={'alpha':0.6})
    plt.title(f'{col} vs Calories')

    # Set x-axis limits based on actual data
    col_min, col_max = df[col].min(), df[col].max()
    plt.xlim(col_min, col_max)

    # Set y-axis limits based on Calories data
    cal_min, cal_max = df['Calories'].min(), df['Calories'].max()
    plt.ylim(cal_min, cal_max)

plt.tight_layout()
plt.show()