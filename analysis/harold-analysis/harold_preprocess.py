"""
You may also normalize or standardize your data in preparation for the next stage, which is ML modeling.
When handling categorical data, you might need to encode them into numerical values.
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import datetime as dt

def make_id_column(list_with_repeats):
    id_format = "21-AXXXX"
    account_dictionary = dict()
    account_dictionary_length = 0
    account_ids = []
    for account_name in list_with_repeats:
        if account_name not in account_dictionary:
            account_dictionary_length += 1
            new_id = str(account_dictionary_length)
            new_id = "0" * (4 - len(new_id)) + new_id
            account_dictionary[account_name] = new_id
        account_ids.append(id_format[:-4] + account_dictionary[account_name])
    return account_ids

def months_after_election(join_date_str):
    election_dt = dt.date(2022, 5, 10)
    join_y, join_m, join_d = join_date_str.split(" ")[0].split("-")
    join_dt = dt.date(int(join_y), int(join_m), int(join_d))
    diff = join_dt - election_dt
    return diff.days * 12 // 365

df_misinfo = pd.read_csv("../../data/final_final_misinfo.csv")
df_misinfo["account_id"] = make_id_column(list(df_misinfo["account_handle"]))
df_misinfo["diff_joined_election"] = [
    months_after_election(join_date) for join_date in list(df_misinfo["joined"])
]
scaler = StandardScaler()
df_misinfo["diff_joined_election_scaled"] = scaler.fit_transform(
    df_misinfo[["diff_joined_election"]]
)

for item in list(df_misinfo["diff_joined_election_scaled"]):
    print(item)
