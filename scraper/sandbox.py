# pyright: basic
from scraper_engine import ScraperEngine

scraper = ScraperEngine()

tweet_urls = [
    "https://twitter.com/TheInvisibleDon/status/1477871478284566528",
    "https://twitter.com/TheInvisibleDon/status/1477871478284566528",
]
tweet_ids = list(map(lambda tweet_url: tweet_url.split("/")[-1], tweet_urls))

for tweet_id in tweet_ids:
    print(scraper.get_tweet_info(tweet_id, ""))
