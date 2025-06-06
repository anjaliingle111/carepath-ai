import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.replace('?', pd.NA, inplace=True)
    df.dropna(axis=1, thresh=int(0.8 * len(df)), inplace=True)  # Drop cols with >20% missing
    df.drop(['weight', 'payer_code', 'medical_specialty', 'encounter_id', 'patient_nbr'], axis=1, errors='ignore', inplace=True)
    df = df.drop_duplicates()
    return df

def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x in ['<30'] else 0)

    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != 'readmitted']

    le = LabelEncoder()
    for col in categorical_cols:
        try:
            df[col] = le.fit_transform(df[col].astype(str))
        except:
            print(f"Error encoding column: {col}")
    return df

def split_data(df: pd.DataFrame):
    X = df.drop('readmitted', axis=1)
    y = df['readmitted']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def preprocess_pipeline(path: str):
    df = load_data(path)
    df = clean_data(df)
    df = encode_data(df)
    return split_data(df)
