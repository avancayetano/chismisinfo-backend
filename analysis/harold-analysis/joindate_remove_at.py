"""
Pythonic 1, 2, 3a
"""


def months_after_election(join_date_str):
    election_month = 5
    election_year = 22
    month, year = join_date_str.split("/")

    year_difference = int(year) - election_year
    month_difference = int(month) - election_month + year_difference * 12

    return month_difference


def remove_replied_to(tweet_str):
    tweet_words = tweet_str.split()
    number_of_accounts_replied_to = 0
    for word in tweet_words:
        if word[0] == "@":
            number_of_accounts_replied_to += 1
        else:
            break

    return " ".join(tweet_words[number_of_accounts_replied_to:])


def make_id_column(list_with_repeats):
    id_format = "G21-AXXXX"
    account_dictionary = dict()
    account_dictionary_length = 0
    account_ids = []
    for account_name in list_with_repeats:
        if account_name not in account_dictionary:
            account_dictionary_length += 1
            new_id = str(account_dictionary_length)
            new_id = "0" * (4 - len(new_id)) + new_id
            account_dictionary[account_name] = new_id
        account_ids.append(id_format[:5] + account_dictionary[account_name])

    return account_ids


if __name__ == "__main__":
    print(months_after_election("4/13"))

    print(
        remove_replied_to(
            """@quency_martinaC @HLingayan @paosese @T_P_S_H @danwaytoofar @ArielHalili3 @mackkoi5 @Stariray0331 @MaySaltikSiLeni @CBot2014 @Vuddoll2 @emingko8 @AllEspant @RonaldImperial4 @molavefinds @Sakristan1914 @ejo_vend @07Red2 @0gagKaKamPuKiss @CdokSembrana @lino_iam @AntonioMoralid1 @TanginaNewpo @Magbubukid4 @Sanny_2021 @WaAkoPakels101 @lav2heart14 @lez_nootnoot @denceasar @ninabeIlatrix @BalDugho @SexymhieQ @LugawNaLutang @ApeloRoger @NileSneel @docvenil @grandeioze @nguimiii @MickNarvadez @EyeEres @crissandymark @myzteryosa15 @maanabrenica @kalboakopakimo @lenirobredo @pnppio @CIDG_PNP @NBI_OFFICIAL_PH @Vincentiments__ Balikan natin about sa sinasabi mo sa sex video 
Hinahanap ko lang naman ung post ng anak ni leni 

September 14,2016

inaamin niya na sya yon sa post niya! Pero binura niya ang post niya! May nka ss 

Simply lng sinabi niya—- I swear to the HOR it was me “."""
        )
    )

    print(make_id_column([1, 1, 2, 3, 4, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13]))
