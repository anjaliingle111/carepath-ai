from src.eda import plot_gender_distribution, plot_correlation
import pandas as pd

def main():
    df = pd.read_csv("data/diabetic_data.csv")
    plot_gender_distribution(df)
    plot_correlation(df)

if __name__ == "__main__":
    main()

