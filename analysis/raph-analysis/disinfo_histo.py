import numpy as np
import scipy as sp
import random
import pandas as pd
import datetime
from matplotlib import pyplot

import matplotlib.pyplot as plt
import seaborn as sns

url='../../data/master_misinfo.csv'
url2='../../data/analysis-data/factual_tweets_scraped.csv'
url3='../../data/analysis-data/misinfo_tweets_labeled.csv'
df = pd.read_csv(url)
df2 = pd.read_csv(url2)
df3 = pd.read_csv(url2)

df = df.drop_duplicates(subset = "account_handle")
df2 = df2.drop_duplicates(subset = "account_handle")


df['joined'] = pd.to_datetime(df['joined'])
df2['joined'] = pd.to_datetime(df2['joined'])
df3['date_posted'] = pd.to_datetime(df3['date_posted'])

dates = [datetime.date(da.year, da.month, da.day) for da in df['joined']]
dates2 = [datetime.date(da.year, da.month, da.day) for da in df2['joined']]
dates3 = [datetime.date(da.year, da.month, da.day) for da in df3['date_posted']]

print(len(dates))
print(len(dates2))
print(len(dates3))

days = []
days2 = []

for d in range(len(dates)):
    a = datetime.date(2022, 5, 9) - dates[d]
    days.append(a.days)

for d in range(len(dates2)):
    a2 = datetime.date(2022, 5, 9) - dates2[d]
    days2.append(a2.days)

values, counts = np.unique(days, return_counts=True)
values2, counts2 = np.unique(days2, return_counts=True)

data = pd.DataFrame(values, counts)
data2 = pd.DataFrame(values2, counts2)


disinfo = []
for i in range(len(data)):
    disinfo.append("Disinformation")
for i in range(len(data2)):
    disinfo.append("Not Disinformation")

frames = [data, data2]
result = pd.concat(frames)

result['Type'] = disinfo

fig, ax = plt.subplots(1, 1, figsize=(15,5), dpi=120)

sns.histplot(data = days,  bins=30, label='Disinfo', alpha=.7, color='red')
sns.histplot(data = days2, bins=30, label="Not Disinfo", alpha=.7, edgecolor='black', color='yellow')
plt.xlabel('Days Before Election', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title("Accounts Joined Trend", fontsize=30, pad=20)
plt.legend()
 
# Showing the plot using plt.show()
plt.show()