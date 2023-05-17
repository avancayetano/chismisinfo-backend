import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
np.random.seed(100)
# Normality test
from scipy.stats import shapiro

df = pd.read_csv("candidate_misinfo_tweets_edited.csv")
print(df.head(10))

following = df['Following']
followers = df['Followers']

# Plot distributions
sns.histplot(following, color="gray", label='Following', alpha=0.5)
sns.histplot(followers, color="green", label='Followers', alpha=0.5)

plt.xlabel('Following and Followers', fontsize=4)
plt.ylabel('Accounts Count', fontsize=4)
plt.xticks(fontsize = 4)
plt.yticks(fontsize = 4)
plt.title('Histograms of Age for Survivors and Non-Survivors', fontsize=18, pad=20)
plt.legend()
plt.show()