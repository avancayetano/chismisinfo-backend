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

df = pd.read_csv("universal_tweets_unique.csv")
print(df['date_posted'])
df['date_posted'] = pd.to_datetime(df['date_posted'])
print(df['date_posted'])