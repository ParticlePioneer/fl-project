import pandas as pd # type: ignore[import-not-found]
import sys
from pathlib import Path
import subprocess

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

# Splitting the 2 Million Row Dataset into Chunks of 500K
# Dropping Unnecessary Columns with almost 100% missing data

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
    chunk.to_csv("data/dataset_2.csv", index=False, mode='a', header=write_header)

df = pd.read_csv('data/dataset_2.csv', low_memory=False)

print("Dataset loaded. Starting cleaning and preprocessing...")

print("Initial dataset shape:", df.shape)


# Above is cleared twice Below we do row cleaning   

df_cleaned = df.dropna(thresh=30)
print(f"{abs(df['id'].count() - df_cleaned['id'].count())} rows dropped due to missing values.")

# 22230 Rows dropped here due to missing values. We can drop these rows as they are not significant in number.
# df_cleaned.to_csv("reduced_row_data.csv", index=False)

df = df_cleaned
df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")

approved_loans = df[
    (df["issue_d"] >= "2015-01-01") &
    (df["issue_d"] <= "2018-12-31")
]
approved_loans.info()
approved_loans["earliest_cr_line"] = pd.to_datetime(
    approved_loans["earliest_cr_line"],
    format="%b-%Y",
    errors="coerce"
)
approved_loans["earliest_cr_line"].dtype 
approved_loans.info()
approved_loans
new_row_clean = approved_loans.dropna(thresh=30)

print(f"{abs(new_row_clean['id'].count() - approved_loans['id'].count())} rows dropped due to missing values.")

# new_row_clean.to_csv("final_cleaned_data_4.csv", index=False)

df = new_row_clean

# Decisions.md Required Changes in Section
# df = pd.read_csv('final_cleaned_data_4.csv', low_memory=False)
print("Starting final preprocessing...")
df['term'] = df['term'].str.replace(' months', '').astype(int)

# Converted the employement length of 10+ years to 10 and < 1 to 0

df['emp_length'] = df['emp_length'].str.replace(' years', '').str.replace(' year', '').str.replace('< 1', '0').str.replace('10+', '10').astype(int)
df.info()
df['issue_d'] = pd.to_datetime(df["issue_d"], format="%Y-%m-%d", errors="coerce")
df['earliest_cr_line'] = pd.to_datetime(df["earliest_cr_line"], format="%Y-%m-%d", errors="coerce")
df['credit_history_months'] = (
    (df["issue_d"] - df["earliest_cr_line"]).dt.days / 30.44
).round()
df.drop(columns=["earliest_cr_line"], inplace=True)
df["fico_score"] = (
    df["fico_range_low"] + df["fico_range_high"]
) / 2
df.drop(
    columns=["fico_range_low", "fico_range_high"],
    inplace=True
)
df.drop(columns=["grade"], inplace=True)
cols = ['mort_acc', 'pub_rec_bankruptcies', 'revol_util', 'dti']
for col in cols:
    df[col] = df[col].fillna(df[col].median())
x = df # Recover df from here

valid_statuses = {
    "Charged Off": 1,
    "Default": 1,
    "Fully Paid": 0
}

df["default"] = df["loan_status"].map(valid_statuses)

df = df[df["default"].notna()].copy()

df["default"] = df["default"].astype(int)

print(df["default"].value_counts())
print(df["default"].value_counts(normalize=True) * 100)
print(df.groupby("issue_d")["default"].agg(
    loans="count",
    default_rate="mean"
))
df["issue_year"] = df["issue_d"].dt.year
print(
    df.groupby("issue_year")["default"].agg(
        loans="count",
        default_rate="mean"
    )
)
df.to_csv("data/processed/synthetic_lendingclub.csv",index=False)
# df = pd.read_csv('decisions_implementation_5.csv', low_memory=False)
# df.info()
print("Done! The processed dataset has been saved to 'data/processed/synthetic_lendingclub.csv'.")