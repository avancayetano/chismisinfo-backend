from datetime import datetime
import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
import calendar


df = pd.read_csv("../../../data/analysis-data/misinfo_tweets_labeled.csv")

leni = np.array(df["leni_sentiment"])
marcos = np.array(df["marcos_sentiment"])

l_count = []
m_count = []

m_s = ['Negative', 'Neutral', 'Positive']
l_s = ['Negative', 'Neutral', 'Positive']

for i in ['Negative', 'Neutral', 'Positive']:
    l_count.append(np.count_nonzero(leni == i))    
    m_count.append(np.count_nonzero(marcos == i))   
  
N = 2
ind = np.arange(N) 
width = 0.25
  
xvals = [l_count[0], m_count[0]]
bar1 = plt.bar(ind, xvals, width, color = 'b')
  
yvals = [l_count[1], m_count[1]]
bar2 = plt.bar(ind+width, yvals, width, color='orange')
  
zvals = [l_count[2], m_count[2]]
bar3 = plt.bar(ind+width*2, zvals, width, color = 'g')
  
plt.xlabel("Sentiments")
plt.ylabel('Count')
plt.title("Number of Tweets by Sentiment")
  
plt.xticks(ind+width,['Leni-Sentiment', 'Marcos-Sentiment'])
plt.legend( (bar1, bar2, bar3), ('Negative', 'Neutral', 'Positive') )
plt.show()

#----









