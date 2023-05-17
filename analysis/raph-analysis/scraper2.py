import snscrape.modules.twitter as sntwitter
import pandas as pd
import datetime

# Creating list to append tweet data to
tweets_list2 = []
search_term = "bbm wharton"

years = [
    "2016",
    "2017",
    "2018",
    "2019",
    "2020",
    "2021",
    "2022"
]

months = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12"
]

end_of_month_days = [
    "31",
    "28",
    "31",
    "30",
    "31",
    "30",
    "31",
    "31",
    "30",
    "31",
    "30",
    "31"
]

# Using TwitterSearchScraper to scrape data and append tweets to list

for year in years:
    for month in range(len(months)):
        print(f"Starting with {year}-{months[month]}-01")
        for i, tweet in enumerate(
                sntwitter.TwitterSearchScraper(f'{search_term} since:{year}-{months[month]}-01 until:{year}-{months[month]}-{end_of_month_days[month]}').get_items()):
            if i >= 300:
                break
            tweets_list2.append([tweet.date,
                                 tweet.id,
                                 tweet.rawContent,
                                 tweet.replyCount,
                                 tweet.likeCount,
                                 tweet.retweetCount,
                                 tweet.quoteCount,
                                 tweet.url,
                                 " ",
                                 tweet.user.username,
                                 tweet.user.displayname,
                                 tweet.user.rawDescription,
                                 tweet.user.renderedDescription,
                                 tweet.user.verified,
                                 tweet.user.created,
                                 tweet.user.followersCount])
        print(f"Done with {year}-{months[month]}-{end_of_month_days[month]}")

# Creating a dataframe from the tweets list above
tweets_df2 = pd.DataFrame(tweets_list2, columns=['Datetime',
                                                 'Tweet Id',
                                                 'Text',
                                                 'Replies',
                                                 'Likes',
                                                 'Retweets',
                                                 'Quote Retweets',
                                                 'URL',
                                                 'Intentionally left blank',
                                                 'Username',
                                                 'Display Name',
                                                 'Raw Description',
                                                 'Rendered Description',
                                                 'Verified?',
                                                 'Date Created',
                                                 'Followers Count'])

tweets_df2.to_csv(f'term_{search_term} date_{datetime.date.today()}.csv')

print(tweets_df2['Text'])