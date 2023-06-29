import numpy as np
import scipy as sp
import pandas as pd
import datetime

import matplotlib.pyplot as plt
import seaborn as sns

url='../../data/master_misinfo.csv'
df = pd.read_csv(url)

df['joined'] = pd.to_datetime(df['joined'])
dates = [datetime.date(da.year, da.month, da.day) for da in df['joined']]

days = []

for d in range(len(dates)):
    a = datetime.date(2022, 5, 9) - dates[d]
    days.append(a.days)

values, counts = np.unique(days, return_counts=True)

print(values)
print(counts)

data = pd.DataFrame(values, counts)

print(data.head())


sns.histplot(data, x = values, bins = 20)
# Format plot
plt.ticklabel_format(style='plain', axis='y')
plt.xticks(rotation=45, size=5)
plt.xlabel('Date')
plt.ylabel('Number of Accounts')
plt.title('Trend of Twitter Accounts Joined', size=18, y=1.02)

# Show the plot
plt.show()