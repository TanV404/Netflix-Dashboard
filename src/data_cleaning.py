import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path)
    df.drop_duplicates(inplace=True)
    df.fillna({'country':'Unknown'}, inplace=True)
    df.fillna({'rating':'Not Rated'}, inplace=True)
    df.fillna({'director':'No Director'}, inplace=True)
    df.fillna({'cast':'No Cast Info'}, inplace=True)
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['listed_in'] = df['listed_in'].astype(str)
    return df
