import pandas as pd
import sys
from pathlib import Path
import subprocess
# Downloading dataset
RAW_DIR = Path("data/raw")
RAW_FILE = RAW_DIR / "accepted_2007_to_2018Q4.csv.gz"

RAW_DIR.mkdir(parents=True, exist_ok=True)

if not RAW_FILE.exists():
    print("Downloading LendingClub dataset from Kaggle...")

    subprocess.run([
        sys.executable,
        "-m", "kaggle",
        "datasets",
        "download",
        "wordsforthewise/lending-club",
        "-f",
        "accepted_2007_to_2018Q4.csv.gz",
        "-p",
        str(RAW_DIR)
    ], check=True)

print("Dataset found.")
print("Starting preprocessing...")
headers =  [    'id','loan_amnt',
                'term','int_rate',
                'installment','grade',
                'sub_grade','emp_length',
                'home_ownership','annual_inc',
                'verification_status','issue_d',
                'loan_status','purpose',
                'addr_state','dti',
                'delinq_2yrs','earliest_cr_line',
                'fico_range_low','fico_range_high',
                'inq_last_6mths','open_acc',
                'pub_rec','revol_bal',
                'revol_util','total_acc',
                'tot_cur_bal','bc_util',
                'mort_acc','pub_rec_bankruptcies'
            ]


for i, chunk in enumerate(pd.read_csv(RAW_FILE,
                  usecols=headers, low_memory=False, chunksize=500000)):
    print(f"Processing chunk {i+1}/5 with {len(chunk)} rows...")
    write_header = (i == 0)
    chunk.to_csv("data/dataset.csv", index=False, mode='a', header=write_header)

df = pd.read_csv('data/dataset.csv', low_memory=False)

print("Dataset loaded as csv. Starting cleaning and preprocessing...")
print("Initial dataset shape:", df.shape)

print("\nImputing rows from a given subset")
subset = [
            'id','loan_amnt',
            'term','int_rate',
            'installment','grade',
            'sub_grade','home_ownership',
            'annual_inc','verification_status',
            'loan_status','purpose',
            'addr_state','earliest_cr_line',
            'fico_range_low','fico_range_high'
]
before = len(df)
df.dropna(subset=subset, inplace=True)
after = len(df)
print(f"Dropped {before - after} rows with missing values in subset columns.")

print("Type cast and limit for 1-1-2015 to 31-12-2018")
df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
df = df[
    (df["issue_d"] >= "2015-01-01") &
    (df["issue_d"] <= "2018-12-31")
]
df["earliest_cr_line"] = pd.to_datetime(
    df["earliest_cr_line"],
    format="%b-%Y",
    errors="coerce"
)

print(f"\nBefore Cleanup : {df.shape}\nFinal Cleanup and Preprocessing Steps:")
df['term'] = df['term'].str.replace(' months', '').astype(int)
df['credit_history_months'] = (
    (df["issue_d"] - df["earliest_cr_line"]).dt.days / 30.44
).round()
df["fico_score"] = (
    df["fico_range_low"] + df["fico_range_high"]
) / 2
print("Dropping columns: 'earliest_cr_line', 'fico_range_low', 'fico_range_high', 'grade'")
df.drop(columns=["earliest_cr_line"], inplace=True)
df.drop(
    columns=["fico_range_low", "fico_range_high"],
    inplace=True
)
df.drop(columns=["grade"], inplace=True)

# Filling null values for 'mort_acc', 'pub_rec_bankruptcies', 'revol_util', and 'dti' with median values
cols = ['mort_acc', 'pub_rec_bankruptcies', 'revol_util', 'dti']
for col in cols:
    df[col] = df[col].fillna(df[col].median())

# Mapped loan_status to binary values: 1 for default/Charged Off loans, 0 for fully paid loans
valid_statuses = {
    "Charged Off": 1,
    "Default": 1,
    "Fully Paid": 0
}
df["default"] = df["loan_status"].map(valid_statuses)

# Dropping rows where 'loan_status' is not in valid_statuses
# i.e Current, In Grace Period, Late (16-30 days), Late (31-120 days), "does not meet credit policy"
df = df[df["default"].notna()].copy()
df["default"] = df["default"].astype(int)
for col in ['tot_cur_bal', 'bc_util']:
    null_rate = df[col].isnull().mean() * 100
    print(f"Null Rate for {col}: {null_rate:.2f}%")

# Converting emp_length to numeric values    
df['emp_length'] = (
    df['emp_length']
    .str.replace(' years', '')
    .str.replace(' year', '')
    .str.replace('< 1', '0')
    .str.replace('10+', '10')
    .fillna('-1')
    .astype(int)
)
file_path = Path("data/processed/synthetic_lendingclub.csv")
file_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(file_path, index=False)
print("\nFile saved to data/processed/synthetic_lendingclub.csv")
df.shape
print("\nClass Balance: ")
df['issue_year'] = df['issue_d'].dt.year
print(df['issue_year'].value_counts().sort_index())
print("Final dataset shape:", df.shape)