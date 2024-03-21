import random


POSITIVE_QUERIES = [
    "Keep the {}",
    "Enhance the {}",
    "Amplify the {}",
    "Boost the {}",
    "Increase the {}",
    "Focus on the {}",
]

NEGATIVE_QUERIES = [
    "Remove the {}",
    "Filter out the {}",
    "Reduce the {}",
]

MIXED_QUERIES = []

# TODO: do the inverse as well (negative first)
for positive_query in POSITIVE_QUERIES:
    for negative_query in NEGATIVE_QUERIES:
        connectors = [" and ", ", "]
        connector = random.choice(connectors)
        MIXED_QUERIES.append(positive_query + connector + negative_query.lower())

DESCIPTIVE_QUERIES = [
    "An excerpt of {}",
    "A clip of {}",
    "This is an audio of {}",
    "A clip playing an audio of {}"
]

COMMAND_TYPES = [
    "positive",
    "descriptive",
    "negative",
    "mixed",
]


def caption_to_random_command(caption: str, interferer_captions: list[str]):
    # Sometimes AudioSet's captions have synonyms separated by commas.
    # For example, one class is Accelerating, revving, vroom
    # We can split these and choose one randomly.
    
    if len(interferer_captions) == 0 or len(interferer_captions) > 1:
        raise ValueError("Only a single interference caption is supported for now.")
    else:
        interferer_captions = interferer_captions[0]
    
    def _parse_text(text: str):
        return random.choice(text.split(",")).lower().strip()
    
    caption = _parse_text(caption)
    interferer_captions = [
        _parse_text(cap) for cap in interferer_captions]

    command_type = random.choice(COMMAND_TYPES)
    if command_type == "positive":
        query = random.choice(POSITIVE_QUERIES).format(caption)
    elif command_type == "descriptive":
        query = random.choice(DESCIPTIVE_QUERIES).format(caption)
    elif command_type == "negative":
        query = random.choice(NEGATIVE_QUERIES).format(
            interferer_captions)
    elif command_type == "mixed":
        query = random.choice(MIXED_QUERIES).format(
            caption, interferer_captions)
    else:
        raise ValueError("Invalid command type")

    return query

if __name__ == "__main__":
    caption = "male speech"
    print(caption_to_random_command(caption))
