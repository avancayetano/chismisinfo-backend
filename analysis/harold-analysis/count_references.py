import pandas as pd

vect_df = pd.read_csv("vectorized_labeled_misinfo.csv")

entities = {
    "Aika Robredo": ["aika", "aika diri", "aika robredo", "aika rob", "she admitted"],
    "Bam Aquino": ["bembem"],
    "Bongbong Marcos": ["bbm", "bbmarcos", "marcos"],
    "Gwyneth Chua": ["chua"],
    "Jillian Robredo": [
        "jillian robredo",
        "mrs robredo daughter",
        "hindot ka",
        "jillian",
        "jillrobredo",
        "ma am jill",
    ],
    "Leni Robredo": [
        "kaylenipataytayo",
        "kaylenitalo",
        "leni lugaw",
        "leni robredog",
        "lutangina",
        "mrs robredo",
        "president leni",
        "president leni robredo",
        "vp leni",
        "vice president",
        "withdrawleni",
        "fake vp",
        "fake vp leni",
        "her mom",
        # "len 2x",
        "lenlen",
        "lenlenloser",
        "leni",
        "leni robredo",
        "lenirobredo",
        "lugaw",
        "lutang",
        "lutang ina",
        "lutang ina mo",
        "mama",
        "mama nyo",
        "mom",
        "mother",
        "nanay kong lutang",
        "nanay mong lumulutang",
        "philippines vice president",
        "robredog",
        "saint inamo",
        "sarili niyang ina",
    ],
    "Tricia Robredo": [
        "tricia",
        "tricia robredo",
        "trisha",
        "trisha robredo",
        "vice president daughter",
        "she went straight",
    ],
    "Thinking Pinoy": ["tp"],
    "BBM Supporters": ["bbm supporters", "maka bbm tao"],
    "Communists": ["cpp", "cpp ndf npa", "komunista"],
    "Filipino People": [
        "igorot sa baguio",
        "igorots",
        "igorot people",
        "igorot",
        "igorot friends",
        "igorot native",
        "ilocano",
        "kpatid na igorot",
        "locals",
        "taong bayan",
        "they are good",
        "they are respectful",
        "they value education",
    ],
    "Jillian's Baguio Group": [
        "grupo ni jillian",
        "her camp",
        "her crowd",
        "team nila jillian",
    ],
    "Kakampinks": [
        "baguio fenks",
        "dilapinks",
        "dilawkadiri",
        "dilawan",
        "fenks",
        "kakampikon",
        "kakampwet",
        "kakamdogs",
        "kakampink",
        "kakampinks",
        "kampo ni leni",
        "pink",
        "pinkilawan",
        "pinklawan",
        "supporters nyoga bastos",
    ],
    "Robredo Family": [
        "anak ni leni",
        "anak mo lenlen",
        "anak ni lenlen",
        "anak ni robredo",
        "daughter of robredo",
        "daughter of saint",
        "daughter of lugaw",
        "mga robredo",
        "mga anak niya",
        "robredo",
        "tatlong anak",
    ],
}


def count_report_for(entity: str) -> None:
    total_refs = 0
    counts = []

    print(f"For {entity}:")
    for phrase in entities[entity]:
        count = vect_df[phrase].sum()
        counts.append(count)
        print(f"{phrase} = {count}")
    total_refs = sum(counts)
    print(f"Total ({entity}) = {total_refs}")


for entity in entities:
    count_report_for(entity)
