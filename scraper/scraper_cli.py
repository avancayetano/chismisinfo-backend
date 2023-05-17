# pyright: basic

import argparse
import os
import pandas as pd
from typing import Optional, List
from scraper_engine import ScraperEngine
from data_types import DatasheetRow


class ScraperCLI(ScraperEngine):
    """
    The interface to interact with the scraper.
    Automatically creates the datasheet via commandline.

    """

    def __init__(self, mode: str, collector: str, keywords: str) -> None:
        super().__init__()

        if mode not in ["collect", "clean", "collect-clean"]:
            print("Invalid mode!")
            print("Valid options: 'collect' or 'clean' or 'collect-clean'")
            return

        collector_name: Optional[str] = {
            "avan": "Cayetano, Anthony Van",
            "harold": "Antonio, Harold",
            "raph": "Portuguez, Raphael Justin",
        }.get(collector, None)

        if collector_name is None:
            print("Invalid collector!")
            print("Valid options: 'avan' or 'harold' or 'raph'")
            return

        self.mode = mode
        self.collector = collector
        self.collector_name = collector_name
        self.keywords = keywords
        self.sub_directory = f"../data/{self.collector}-datasets"

    def collect(self) -> None:
        data_list = self.scrape(self.keywords)
        if len(data_list) > 0:
            df_data_list = pd.DataFrame(data_list)
            if os.path.exists(f"{self.sub_directory}/raw_dataset.csv"):
                raw_tweet_urls = set(
                    pd.read_csv(f"{self.sub_directory}/raw_dataset.csv")[
                        "tweet_url"
                    ].to_list()
                )
                df_data_list = df_data_list[
                    ~df_data_list["tweet_url"].isin(raw_tweet_urls)
                ]
            if df_data_list.shape[0] > 0:
                self.df_to_csv(df_data_list, f"{self.sub_directory}/raw_dataset.csv")
            else:
                print("\nNo new tweets scraped!")
        else:
            print("\nNo new tweets scraped!")

    def clean(self) -> None:
        df: pd.DataFrame = pd.read_csv(
            f"{self.sub_directory}/raw_dataset.csv",
            parse_dates=["joined", "date_posted"],
        )
        clean_tweet_urls: set[str] = set([url for url in self.existing_clean_dataset])
        if os.path.exists(f"{self.sub_directory}/clean_dataset.csv"):
            clean_tweet_urls: set[str] = set(
                [
                    *clean_tweet_urls,
                    *pd.read_csv(f"{self.sub_directory}/clean_dataset.csv")[
                        "tweet_url"
                    ].to_list(),
                ]
            )

        clean_datasheet_rows: List[DatasheetRow] = []

        for _, row in df.iterrows():
            if row["tweet_url"] not in clean_tweet_urls:
                datasheet_row = self.tweet_data_to_datasheet_row(
                    row, self.collector_name
                )
                clean_datasheet_rows.append(datasheet_row)

        if len(clean_datasheet_rows) > 0:
            df_clean_datasheet_rows = pd.DataFrame(clean_datasheet_rows)
            self.df_to_csv(
                df_clean_datasheet_rows, f"{self.sub_directory}/clean_dataset.csv"
            )
        else:
            print("No new clean datasheet row added!")

    def execute(self) -> None:
        if self.mode == "collect":
            self.collect()
        elif self.mode == "clean":
            self.clean()
        elif self.mode == "collect-clean":
            self.collect()
            self.clean()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "mode", metavar="Mode", help="Mode - 'collect' or 'clean' or 'collect-clean'"
    )
    argparser.add_argument("-c", help="Collector - 'avan', 'harold' or 'raph'")
    argparser.add_argument("-k", help="Keywords", default=None)

    args = argparser.parse_args()

    mode = args.mode
    collector = args.c
    keywords = args.k

    print("Running command...")
    print(f"Mode: '{mode}' | Collector: '{collector}' | Keywords: '{keywords}'\n")

    """
    Commands:
    > To collect:
    python scraper_cli.py collect -c avan -k "let me educate you robredo"

    > To clean:
    python scraper_cli.py clean -c avan

    > To collect then clean:
    python scraper_cli.py collect-clean -c avan -k "let me educate you robredo"

    """

    scraper_cli = ScraperCLI(mode, collector, keywords)
    scraper_cli.execute()
