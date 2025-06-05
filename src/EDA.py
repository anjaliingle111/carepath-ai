import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_gender_distribution(df, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.countplot(x='gender', data=df)
    plt.title('Gender Distribution')
    plt.xticks(rotation=45)
    plt.savefig(f"{output_dir}/gender_distribution.png")
    plt.close()

def plot_correlation(df, output_dir="outputs"):
    corr = df.select_dtypes(include='number').corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap")
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()

