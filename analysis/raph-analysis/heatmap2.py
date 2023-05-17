import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("candidate_misinfo_tweets_edited.csv")
data = pd.DataFrame(df, columns = ['Likes', 'Replies', 'Retweets', 'Quote Tweets'])

fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(data.corr(), center=0, cmap='jet')
ax.set_title('Correlation Between Likes, Replies, Retweets, and Quote Tweets')
plt.show()