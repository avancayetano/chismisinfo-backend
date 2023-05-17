from datetime import datetime
import pandas as pd 
import numpy as np
import scipy as sp
import seaborn as sns 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
import calendar

df = pd.read_csv("candidate_misinfo_tweets_edited.csv")
print(df.head(10))

# Plot the histogram of the age distribution (training set only)
ax = sns.histplot(data=df, x='Likes', stat='density', label='samples', binwidth=5)

# Compute the probability density function for the normal distribution
# based on the mean and std
mean_likes = df['Likes'].mean()
std_likes = df['Likes'].std()

x0, x1 = ax.get_xlim()                # extract the endpoints for the x-axis
norm_age_x = np.linspace(0, x1, 100)       # set lowest x to 0
norm_age_pdf = sp.stats.norm.pdf(norm_age_x, loc=mean_likes, scale=std_likes) # compute the pdf

# Plot the normal pdf 
ax.plot(norm_age_x, norm_age_pdf, 'r', lw=2, label='pdf')                                                   

plt.title("Likes Distribution vs. Normal Distribution", fontsize=18, pad=20)
ax.legend()

plt.show()