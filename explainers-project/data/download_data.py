import os
import pandas as pd
import urllib.request
import zipfile

ADULT_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
TARGET_DIR = "raw"
OUTPUT_CSV = os.path.join(TARGET_DIR, "adult.csv")

def download_adult():
    os.makedirs(TARGET_DIR, exist_ok=True)
    zip_path = os.path.join(TARGET_DIR, "adult.zip")

    print("Downloading UCI Adult dataset...")
    urllib.request.urlretrieve(ADULT_URL, zip_path)
    print("Downloaded to:", zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(TARGET_DIR)

    os.remove(zip_path)
    print("Extracted to:", TARGET_DIR)

def load_and_clean():
    # The extracted folder contains adult.data and adult.test
    train_path = os.path.join(TARGET_DIR, "adult.data")
    test_path = os.path.join(TARGET_DIR, "adult.test")

    # Column names from UCI
    cols = [
        "age","workclass","fnlwgt","education","education-num","marital-status",
        "occupation","relationship","race","sex","capital-gain","capital-loss",
        "hours-per-week","native-country","income"
    ]

    # Read training data
    df_train = pd.read_csv(train_path, header=None, names=cols, skipinitialspace=True)

    # Read test data (note: last column has a dot)
    df_test = pd.read_csv(test_path, header=None, names=cols, skipinitialspace=True, skiprows=1)
    df_test["income"] = df_test["income"].str.replace(".", "", regex=False)

    df = pd.concat([df_train, df_test], ignore_index=True)

    # Remove rows with missing values represented as "?"
    df = df.replace("?", pd.NA).dropna()

    return df

def save_data(df):
    df.to_csv(OUTPUT_CSV, index=False)
    print("Saved cleaned dataset to:", OUTPUT_CSV)

if __name__ == "__main__":
    download_adult()
    df = load_and_clean()
    save_data(df)
