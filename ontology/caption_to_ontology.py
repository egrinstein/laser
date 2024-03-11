import random


POSITIVE_QUERIES = [
    "Keep the %",
    "Enhance the %",
]

NEGATIVE_QUERIES = [
    "Remove the %",
    "Filter out the %",
    "Reduce the %",
]

MIXED_QUERIES = []

for positive_query in POSITIVE_QUERIES:
    for negative_query in NEGATIVE_QUERIES:
        connectors = ["and ", ", "]
        connector = random.choice(connectors)
        MIXED_QUERIES.append(positive_query + connector + negative_query.lower())


COMMAND_TYPES = [
    "positive",
    "negative",
    "mixed"
]


def caption_to_random_command(caption):
    command_type = random.choice(COMMAND_TYPES)
    if command_type == "positive":
        query = random.choice(POSITIVE_QUERIES)
    elif command_type == "negative":
        query = random.choice(NEGATIVE_QUERIES)
    elif command_type == "mixed":
        query = random.choice(MIXED_QUERIES)
    return query.replace("%", caption)


if __name__ == "__main__":
    caption = "male speech"
    print(caption_to_random_command(caption))