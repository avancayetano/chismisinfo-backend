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

joined_year = []

for d in only_2022['Joined']:
    d = d.split("-")
    if (int(d[0]) < 10):
        year = int('200'+d[0])
    else:
        year = int('20'+d[0])
    joined_year.append(year)

#years = np.array([2017,202022])
#months = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

#for year in years:
#   for month in months:
#        y_m.append((year, month))
#        y_mlabel.append(str((year, month)))

years = np.array([2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022])
count_per_year = []

for y in years:
    count_per_year.append(joined_year.count(y))
    
count_per_year = np.array(count_per_year)

cumulative2 = np.cumsum(count_per_year)
plt.plot(years, cumulative2, c='blue')
plt.show()

data_plot = pd.DataFrame({"Years":years, "Count":count_per_year})
#sns.ecdfplot(data=data_plot, x = "Years", stat = "count")
sns.kdeplot(data=data_plot, x="Years", stacked=True, cumulative = True)
plt.show()

sns.distplot(data=data_plot, x="Years", y="Count")
plt.show()
#df['date_posted'] = pd.to_datetime(df['date_posted'])
#print(df['date_posted'])