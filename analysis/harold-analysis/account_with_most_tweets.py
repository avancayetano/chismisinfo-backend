import pandas as pd

df = pd.read_csv("labeled_candidate_misinfo - filtered.csv")

print(df.account_handle.value_counts())
