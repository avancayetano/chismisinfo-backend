from matplotlib import pyplot as plot
import numpy as np
import seaborn as sns
import pandas as pd
np.random.seed(100)
# Normality test
from scipy.stats import shapiro

df = pd.read_csv("../../data/analysis-data/misinfo_tweets_labeled.csv")
print(df.head(10))

incident = df['incident']
baguio = incident.value_counts()['Baguio']
scandal = incident.value_counts()['Scandal']
quar = incident.value_counts()['Quarantine']
ladder = incident.value_counts()['Ladder']
others = incident.value_counts()['Others']

incidents = ["Baguio", "Scandal", "Quarantine", "Ladder", "Others"]
counts = [baguio, scandal, quar, ladder, others]

data_plot = pd.DataFrame({"Incidents":incidents, "Count":counts})

sns.barplot(data=data_plot, x="Incidents", y="Count", capsize=.4, edgecolor=".5")
plot.title("Number of Tweets Per Incident")
plot.xlabel("Incidents")
plot.ylabel("Number of Tweets")
plot.show()




# Plot distributions


#plt.bar(incident, color="gray", label='Incident', height=0.5)
#plt.xlabel('Following and Followers', fontsize=4)
#plt.ylabel('Accounts Count', fontsize=4)
#plt.xticks(fontsize = 4)
#plt.yticks(fontsize = 4)
#plt.title('Histograms of Age for Survivors and Non-Survivors', fontsize=18, pad=20)
#plt.legend()
#plt.show()