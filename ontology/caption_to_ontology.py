import random


POSITIVE_QUERIES = [
    "Keep the %",
    "Enhance the %",
    "Amplify the %",
    "Boost the %",
    "Increase the %",
    "Focus on the %",
]

NEGATIVE_QUERIES = [
    "Remove the %",
    "Filter out the %",
    "Reduce the %",
]

MIXED_QUERIES = []

# TODO: do the inverse as well (negative first)
for positive_query in POSITIVE_QUERIES:
    for negative_query in NEGATIVE_QUERIES:
        connectors = ["and ", ", "]
        connector = random.choice(connectors)
        MIXED_QUERIES.append(positive_query + connector + negative_query.lower())

DESCIPTIVE_QUERIES = [
    "An excerpt of % and %.",
    "A clip of % and %.",
    "This is an audio of % and %."
    "This clip plays an audio of % and %."
]

COMMAND_TYPES = [
    "positive",
    "descriptive"
    # "negative",
    # "mixed",
]


def caption_to_random_command(caption):
    command_type = random.choice(COMMAND_TYPES)
    if command_type == "positive":
        query = random.choice(POSITIVE_QUERIES)
    elif command_type == "descriptive":
        query = random.choice(DESCIPTIVE_QUERIES)
    elif command_type == "negative":
        raise NotImplementedError("Add negative queries")
        query = random.choice(NEGATIVE_QUERIES)
    elif command_type == "mixed":
        raise NotImplementedError("Add negative queries")
        query = random.choice(MIXED_QUERIES)
    return query.replace("%", caption)


if __name__ == "__main__":
    caption = "male speech"
    print(caption_to_random_command(caption))