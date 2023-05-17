import pandas as pd

"""
agenda
count negative leni         50/203
count positive marcos       16/203

count the sentiments for chi square
negative leni and has reference         47
negative leni and no reference          1
nonnegative leni and has reference      50
nonnegative leni and no reference       105
"""
leni_names = [
    "kaylenipataytayo",
    "kaylenitalo",
    "leni lugaw",
    "leni robredog",
    "lutangina",
    "mrs robredo",
    "president leni",
    "president leni robredo",
    "vp leni",
    "vice president",
    "withdrawleni",
    "fake vp",
    "fake vp leni",
    "her mom",
    # "len 2x",
    "lenlen",
    "lenlenloser",
    "leni",
    "leni robredo",
    "lenirobredo",
    "lugaw",
    "lutang",
    "lutang ina",
    "lutang ina mo",
    # "mananalo si robredo",
    "mama",
    "mama nyo",
    "mom",
    "mother",
    "nanay kong lutang",
    "nanay mong lumulutang",
    "philippines vice president",
    "robredog",
    "saint inamo",
    "sarili niyang ina",
]
df_vect = pd.read_csv("vectorized_labeled_misinfo.csv")
df_labeled = pd.read_csv("misinfo_tweets_labeled.csv")


def has_ref(index):
    return 0 < sum([df_vect.at[index, leni_name] for leni_name in leni_names])


def chi_square_report():
    sentis = list(df_labeled["leni_sentiment"])
    for i in range(203):
        senti = sentis[i]


print(has_ref(202))
