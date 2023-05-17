# pyright: basic

import itertools
import os
import json
import pandas as pd
import snscrape.modules.twitter as sntwitter
from snscrape.modules.twitter import Tweet, Medium, Photo, Video, Gif
from data_types import MediumDict, TweetData, DatasheetRow
from typing import List, Union
from datetime import datetime


class ScraperEngine:
    """
    Scraper engine
    """

    def __init__(self) -> None:
        self.existing_clean_dataset = set()

    def __search_tweets(
        self, keywords: str, start_date: str, end_date: str
    ) -> List[TweetData]:
        return list(
            map(
                lambda tweet: self.__convert_to_tweet_data(tweet, keywords),
                sntwitter.TwitterSearchScraper(
                    f"{keywords} since:{start_date} until:{end_date}"
                ).get_items(),
            )
        )

    def get_tweet_info(self, tweet_id: str, keywords: str) -> TweetData:
        return self.__convert_to_tweet_data(
            list(sntwitter.TwitterTweetScraper(tweet_id).get_items())[0], keywords
        )

    def scrape(self, keywords: str) -> List[TweetData]:
        years = ["2020", "2021", "2022"]
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
            "12",
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
            "31",
        ]

        year_month_day = list(
            map(
                lambda x: (x[0], *x[1]),
                itertools.product(years, zip(months, end_of_month_days)),
            ),
        )
        year_month_day[1] = ("2020", "02", "29")
        tweet_data_list: List[TweetData] = []

        for y, m, d in year_month_day:
            start_date, end_date = f"{y}-{m}-01", f"{y}-{m}-{d}"
            print(f"Searching {start_date} to {end_date}...")
            row = self.__search_tweets(keywords, start_date, end_date)
            tweet_data_list.extend(row)
            print(f"> Found {len(row)} tweets!\n")

        return tweet_data_list

    def get_user_tweets(self, username: str) -> List[TweetData]:
        user_tweets: List[TweetData] = []
        # nested try-except (horrible code oh no...) (im just cramming at this point)
        try:
            for tweet in sntwitter.TwitterProfileScraper(f"{username}").get_items():
                try:
                    user_tweets.append(self.__convert_to_tweet_data(tweet, ""))
                except:
                    pass
        except:
            pass

        return user_tweets

    def __convert_to_tweet_data(self, tweet: Tweet, keywords: str) -> TweetData:
        return {
            "tweet_url": tweet.url,
            "tweet_id": tweet.id,
            "keywords": keywords,  # Keywords
            "account_handle": tweet.user.username,  # Account handle
            "account_name": tweet.user.displayname,  # Account name
            "account_bio": tweet.user.rawDescription,  # Account bio
            "account_bio_rendered": tweet.user.renderedDescription,  # Account bio
            "account_verified": tweet.user.verified,  # Account type
            "joined": tweet.user.created,  # Joined
            "following": tweet.user.friendsCount,  # Following
            "followers": tweet.user.followersCount,  # Followers
            "location": tweet.user.location,  # Location
            "tweet": tweet.rawContent,  # Tweet
            "tweet_rendered": tweet.renderedContent,  # Tweet
            "source_url": tweet.sourceUrl,  # Tweet Alt-text | Tweet Type
            "source_label": tweet.sourceLabel,  # Tweet Alt-text | Tweet Type
            "links_url": list(map(lambda link: link.url, tweet.links))
            if tweet.links
            else None,  # Tweet Alt-text | Tweet Type
            "media": list(map(lambda medium: self.__format_medium(medium), tweet.media))
            if tweet.media
            else None,  # Tweet Alt-text | Tweet Type
            "retweeted_tweet_id": tweet.retweetedTweet.id
            if tweet.retweetedTweet
            else None,  # Tweet Alt-text | Tweet Type
            "quoted_tweet_id": tweet.quotedTweet.id
            if tweet.quotedTweet
            else None,  # Tweet Alt-text | Tweet Type
            "in_reply_to_tweet_id": tweet.inReplyToTweetId,  # Tweet Alt-text | Tweet Type
            "in_reply_to_user_id": tweet.inReplyToUser.id
            if tweet.inReplyToUser
            else None,  # Tweet Alt-text | Tweet Type
            "date_posted": tweet.date,  # Date posted
            "likes": tweet.likeCount,  # Likes
            "replies": tweet.replyCount,  # Replies
            "retweets": tweet.retweetCount,  # Retweets
            "quote_tweets": tweet.quoteCount,  # Quote Tweets
            "views": tweet.viewCount,  # Views
            "conversation_id": tweet.conversationId,
        }

    def __format_medium(self, medium: Medium | Photo | Video | Gif) -> MediumDict:
        medium_type = medium.__class__.__name__.lower()
        if type(medium) == Photo:
            return {
                "type": medium_type,
                "url": medium.fullUrl,
                "alt_text": medium.altText,
            }
        elif type(medium) == Video or type(medium) == Gif:
            return {
                "type": medium_type,
                "url": medium.thumbnailUrl,
                "alt_text": medium.altText,
            }
        else:
            return {"type": medium_type, "url": "", "alt_text": None}

    def df_to_csv(self, df: pd.DataFrame, path: str) -> None:
        print(f"\n>>> Dataframe shape: {df.shape}")
        if not os.path.isfile(path):
            df.to_csv(f"{path}", index=False)
        else:
            df.to_csv(f"{path}", mode="a", index=False, header=False)
        print(f">>> Outputting csv file to {path}\n")

    def collect_all_data(self) -> None:
        """
        Collects all clean data from each data subdirectory, such that
        every row is unique and clean.
        """
        master_dataset_name = "master_dataset.csv"
        print(master_dataset_name)
        pass

    def tweet_data_to_datasheet_row(
        self, tweet_data: pd.Series, collector_name: str
    ) -> DatasheetRow:
        remarks = ""
        tweet_type = "Text " if len(tweet_data["tweet"]) > 0 else ""

        tweet_content = tweet_data["tweet"]

        if pd.notna(tweet_data["links_url"]):
            tweet_type += "URL "

        if pd.isna(tweet_data["account_bio"]):
            remarks += "Account has no bio.\n"

        if pd.isna(tweet_data["location"]):
            remarks += "Account has no location.\n"

        if (
            tweet_data["conversation_id"]
            or tweet_data["in_reply_to_tweet_id"]
            or tweet_data["in_reply_to_user_id"]
            or tweet_data["quoted_tweet_id"]
        ):
            tweet_type += "Reply "
            remarks += (
                "Tweet is a quote tweet.\n"
                if tweet_data["quoted_tweet_id"]
                else "Tweet is part of a thread.\n"
            )

        if pd.notna(tweet_data["media"]):
            media = json.loads(
                tweet_data["media"].replace("'", '"').replace("None", '""')
            )
            for medium in media:
                if (
                    medium["type"] == "photo" or medium["type"] == "gif"
                ) and "photo" not in tweet_type:
                    tweet_type += "Image "
                    tweet_content += "\n{Alt-text image: <insert here...>}"

                if medium["type"] == "video" and "Video" not in tweet_type:
                    tweet_type += "Video "
                    tweet_content += "\n{Alt-text video: <insert here...>}"

        tweet_type = tweet_type.rstrip(" ").replace(" ", ", ")

        return {
            "tweet_url": tweet_data["tweet_url"],
            "group": 21,
            "collector": collector_name,
            "category": "RBRD",
            "topic": "Allegations Against Robredo Sisters",
            "keywords": tweet_data["keywords"],
            "account_handle": tweet_data["account_handle"],
            "account_name": tweet_data["account_name"],
            "account_bio": tweet_data["account_bio"],
            "account_type": None,
            "joined": tweet_data["joined"].strftime("%m/%y")
            if tweet_data["joined"] is not None
            else None,
            "following": tweet_data["following"],
            "followers": tweet_data["followers"],
            "location": tweet_data["location"],
            "tweet": tweet_content,
            "tweet_translated": None,
            "tweet_type": tweet_type,
            "date_posted": tweet_data["date_posted"].strftime("%d/%m/%y %H:%M"),
            "screenshot": None,
            "content_type": None,
            "likes": tweet_data["likes"],
            "replies": tweet_data["replies"],
            "retweets": tweet_data["retweets"],
            "quote_tweets": tweet_data["quote_tweets"],
            "views": tweet_data["views"],
            "rating": None,
            "reasoning": None,
            "remarks": remarks,
        }
