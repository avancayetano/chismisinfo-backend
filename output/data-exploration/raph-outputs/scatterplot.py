from datetime import datetime
import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
import calendar

df = pd.read_csv("../../../data/control_dataset_testing.csv")

baguio = ['let me educate you igorot', 'let me educate you robredo', 'let me educate you jillian', 'let me educate you jilian', 'let me educate you baguio',
'let me educate you bastos', 'let me educate you palengke', 'jillian_baguio_bastos']
scandal = ['robredo_scandal', 'aika robredo sex video', 'aika sex scandal', 'aika inamin', 'leni anak scandal', 'leni anak sex video', 'tricia robredo sex video']
quarantine = ['leni anak quarantine', 'aika quarantine', 'tricia robredo quarantine', 'robredo anak quarantine', 'jay sonza quarantine', 'aika violate quarantine']
ladder = ['tricia hagdan', 'tricia robredo ladder']
others = ['tricia robredo flyers', 'leni anak harvard', 'tricia robredo leaflet']

baguio_df = df[[d in baguio for d in df['keywords']]]
scandal_df = df[[d in scandal for d in df['keywords']]]
quarantine_df = df[[d in quarantine for d in df['keywords']]]
ladder_df = df[[d in ladder for d in df['keywords']]]
others_df = df[[d in others for d in df['keywords']]]

print(others_df)

sns.scatterplot(data=df, x="likes", y="retweets")
plt.title("Scatterplot showing the Correlation Between Likes and Retweets")
plt.show()