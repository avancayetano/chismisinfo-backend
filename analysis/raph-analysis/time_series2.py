# Time Series Analysis
# [ ] Interpolation
# [ ] Binning

from datetime import datetime
import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
import calendar

#only tweets posted in 2022

df = pd.read_csv("candidate_misinfo_tweets_edited.csv")
df['Date posted'] = pd.to_datetime(df['Date posted'])

only_2022 = df[[d.year == 2022 for d in df['Date posted']]]

#print(only_2022['Date posted'])

a = [(d.year, d.month) for d in only_2022['Date posted']] 

years = np.array([2022])
months = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

count_per_month = []
count_per_week = []
count_per_day = []

y_m = []
y_mlabel = []
wlabel = []
dlabel = []

for year in years:
    for month in months:
        y_m.append((year, month))
        y_mlabel.append(str((year, month)))

for i in y_m:
    count_per_month.append(a.count(i))

count_per_month = np.array(count_per_month)

sns.kdeplot(data=y_m)
plt.show()

plt.title("Tweets by Month")
plt.figure(figsize=(8,6), dpi=80)
plt.xlabel('(Year, Month)', fontsize = 10)
plt.xticks(fontsize = 10)
plt.ylabel('Number of Tweets', fontsize = 10)
plt.yticks(fontsize = 10)
plt.bar(y_mlabel , count_per_month, color = "orange")
plt.show()


b = only_2022['Date posted'].dt.isocalendar().week
b = np.array(b)

for i in range(53):
    wlabel.append(i)
    count_per_week.append(np.count_nonzero(b == i))

plt.title("Tweets by Week")
plt.figure(figsize=(8,6), dpi=80)
plt.xlabel('Week Number', fontsize = 10)
plt.xticks(fontsize = 10)
plt.ylabel('Number of Tweets', fontsize = 10)
plt.yticks(fontsize = 10)
plt.bar(wlabel , count_per_week, color = "orange")
plt.show()

c = only_2022['Date posted'].dt.dayofyear
c = np.array(c)
#print(c)

for i in range(366):
    dlabel.append(i)
    count_per_day.append(np.count_nonzero(c == i))

plt.title("Tweets by Day")
plt.figure(figsize=(8,6), dpi=80)
plt.xlabel('Day Number', fontsize = 10)
plt.xticks(fontsize = 10)
plt.ylabel('Number of Tweets', fontsize = 10)
plt.yticks(fontsize = 10)
plt.bar(dlabel , count_per_day, color = "orange")
plt.show()

