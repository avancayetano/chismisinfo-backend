from datetime import datetime
import pandas as pd 
import numpy as np
import plotly.express as px
import seaborn as sns 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
import calendar
import chart_studio
import chart_studio.plotly as py
import chart_studio.tools as tls
import json


df = pd.read_csv("../../../analysis/raph-analysis/candidate_misinfo_tweets_edited.csv")

#only tweets posted in 2022
df['Date posted'] = pd.to_datetime(df['Date posted'])
print(df['Joined'])

joined = []
joined_year = []
joined_year_count = []
joined_month_count = []

for d in df['Joined']:
    d = d.split("-")

    year = " "
    month = " "
    if int(d[0]) < 10:  year = "200"+d[0]
    else:               year = "20"+d[0]
    year = int(year)

    match d[1]:
        case "Jan": month = '01'
        case "Feb": month = '02'
        case "Mar": month = '03'
        case "Apr": month = '04'
        case "May": month = '05'
        case "Jun": month = '06'
        case "Jul": month = '07'
        case "Aug": month = '08'
        case "Sep": month = '09'
        case "Oct": month = '10'
        case "Nov": month = '11'
        case "Dec": month = '12'
        
    joined.append(str(year)+"-"+month)
    joined_year.append(year)

joined_year.sort()

years = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
months2 = []
for y in years:
    joined_year_count.append(joined_year.count(y))
    for m in months:
        joined_month_count.append(joined.count(str(y)+"-"+m))
        months2.append(str(y)+"-"+m)


joined_year_count = np.cumsum(joined_year_count)
joined_month_count = np.cumsum(joined_month_count)

#joined_month_count = np.cumsum(joined_month_count)

df2 = pd.DataFrame({"Months": years, "Count": joined_year_count})

print(df2)

sns.lineplot(data=df2, x="Months", y="Count")
plt.xlabel('Time', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.title("Cumulative Number Of Accounts Joined", fontsize=20, pad=20)
plt.legend()

# Showing the plot using plt.show()
plt.show()