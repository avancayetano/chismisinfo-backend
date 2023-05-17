from typing import Optional, Literal, TypedDict, List
import datetime


class MediumDict(TypedDict):
    type: str
    url: str
    alt_text: Optional[str]


class TweetData(TypedDict):
    """
    All the relevant attributes of Tweet.
    Object attributes are converted to dict of relevant attibutes.
    """

    # Column names specified on the datasheet that are also attributes of Tweet
    tweet_url: str  # Tweet URL
    keywords: str  # Keywords
    account_handle: str  # Account handle
    account_name: Optional[str]  # Account name
    account_bio: Optional[str]  # Account bio

    joined: Optional[datetime.datetime]  # Joined
    following: Optional[int]  # Following
    followers: Optional[int]  # Followers
    location: Optional[str]  # Location

    tweet: str  # Tweet
    date_posted: datetime.datetime  # Date posted
    likes: int  # Likes
    replies: int  # Replies
    retweets: int  # Retweets
    quote_tweets: int  # Quote Tweets
    views: Optional[int]  # Views

    # ---------------------------------------------------------------------------
    # Below are some attributes not included in the datasheet specs
    # but could be of use...

    # Rendered attributes (the raw versions are considered the actual)
    account_bio_rendered: Optional[str]  # Account bio
    tweet_rendered: str  # Tweet

    tweet_id: int  # Useful meta-data

    account_verified: Optional[bool]  # Account type

    source_url: Optional[str]  # Tweet Alt-text | Tweet Type
    source_label: Optional[str]  # Tweet Alt-text | Tweet Type
    links_url: Optional[List[str]]  # Tweet Alt-text | Tweet Type
    media: Optional[List[MediumDict]]  # Tweet Alt-text | Tweet Type
    retweeted_tweet_id: Optional[int]  # Tweet Alt-text | Tweet Type
    quoted_tweet_id: Optional[int]  # Tweet Alt-text | Tweet Type
    in_reply_to_tweet_id: Optional[int]  # Tweet Alt-text | Tweet Type
    in_reply_to_user_id: Optional[int]  # Tweet Alt-text | Tweet Type

    conversation_id: int


class DataRow(TypedDict):
    """
    Data used in analysis
    TODO: this is to be done during the data analysis stage...
    """

    pass


class DatasheetRow(TypedDict):
    """
    A single row in the datasheet specified by Sir.
    All columns except `ID` and `Timestamp`
    """

    tweet_url: str

    group: Literal[21]
    collector: str
    category: Literal["RBRD"]
    topic: Literal["Allegations Against Robredo Sisters"]

    keywords: str
    account_handle: str
    account_name: Optional[str]
    account_bio: Optional[str]
    account_type: Optional[str]
    joined: Optional[str]
    following: Optional[int]
    followers: Optional[int]
    location: Optional[str]

    tweet: str
    tweet_translated: Optional[str]
    tweet_type: str
    date_posted: str
    screenshot: Optional[str]
    content_type: Optional[str]
    likes: int
    replies: int
    retweets: int
    quote_tweets: Optional[int]
    views: Optional[int]
    rating: Optional[str]
    reasoning: Optional[str]

    remarks: Optional[str]
