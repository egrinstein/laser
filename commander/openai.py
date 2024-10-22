import json
import random

from openai import OpenAI

POSITIVE_COMMAND = """
Given a target caption referring to an audio scene,
create a json with a key 'command', whose value is a textual separation command.
Examples of expected verbs are 'Keep', 'Enhance', 'Amplify', 'Boost', 'Increase', 'Focus on' and 'Extract'.
Feel free to simplify the caption to make the command more concise. For example, given a target caption

Target: A woman talks nearby as water pours

Example commands could be:

{"command": Enhance the woman talking and the water pouring}
{"command": Keep the woman talking and the water pouring"}
{"command": Isolate the woman talking nearby and the water pouring"}
"""

NEGATIVE_COMMAND = """
Given an interferer caption referring to an audio scene,
create a json with a key 'command', whose value is a textual separation command.
Examples of expected verbs are 'Remove', 'Filter out', 'Reduce', 'Eliminate' and 'Discard'.
Feel free to simplify the caption to make the command more concise.
For example, given an interferer caption

Interferer: Dishes are clanging

Example output commands could be:

{"command": Remove the clanging dishes}
{"command": Reduce the clanging dishes}
{"command": Exclude the clanging dishes"}
"""

MIXED_COMMAND = """
Given a target and an interferer captions referring to an audio scene,
create a json with a key 'command', whose value is a textual separation command.
Examples of expected verbs related to the Target are 'Keep', 'Enhance', 'Amplify', 'Boost', 'Increase', 'Focus on' and 'Extract',
and for the Interferer are 'Remove', 'Filter out', 'Reduce', 'Eliminate' and 'Discard'.
Feel free to simplify the target and interferer captions to make the command more concise.
For example, given the following target and interferer captions,

Target: A woman talks nearby as water pours

Interferer: Dishes are clanging

Example output commands could be:

{"command": Remove the clanging dishes, and enhance the woman talking and the water pouring}
{"command": Keep the woman talking nearby and the water pouring, and exclude the clanging dishes}
"""

COMMAND = """Given a target and an interferer captions referring to an audio scene,
create a json with three keys, 'positive', 'negative' and 'mixed', each containing a textual separation command.
A separation is itself a json containing
For example:

Target: A woman talks nearby as water pours
Interferer: Dishes are clanging

Example output:
{
    "positive": "Enhance the sound of the woman talking nearby and the water pouring",
    "negative": "Remove the clanging dishes",
    "mixed": "Keep the sound of the woman talking nearby and the water pouring, and exclude the clanging dishes"
}
"""

COMMAND_TYPES = ["mixed", "positive", "negative"]

COMMANDS = {
    "positive": POSITIVE_COMMAND,
    "negative": NEGATIVE_COMMAND,
    "mixed": MIXED_COMMAND
}

def random_openai_command(client, target_caption, interferer_caption, return_type=False):
    command_type = random.choice(COMMAND_TYPES)
    command = COMMANDS[command_type]

    messages = [
        {"role": "system", "content": "You are an AI assistant working in the field of Language-Queried Audio Sound Separation (LASS)."},
        {"role": "user", "content": command},
        {"role": "user", "content": f"Type: {command_type}"},
    ]

    if command_type == "positive" or command_type == "mixed":
        messages.append({"role": "user", "content": f"Target: {target_caption}"})
    if command_type == "negative" or command_type == "mixed":
        messages.append({"role": "user", "content": f"Interferer: {interferer_caption}"})

    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    response_format = { "type": "json_object" }
    ).choices[0].message.content

    try:
        response = json.loads(response)
        command = response["command"]
    except KeyError:
        command = None

    if return_type:
        return command, command_type
    else:
        return command


class OpenAiCommandCreator:
    def __init__(self) -> None:
        self.client = OpenAI()

    def __call__(self, target_caption, interferer_captions, return_type=False):
        return random_openai_command(
            self.client, target_caption, interferer_captions[0],
            return_type=return_type)
        

if __name__ == "__main__":
    client = OpenAiCommandCreator()

    TARGET_CAPTION = "A woman talks nearby as water pours"
    INTERFERER_CAPTION = "A man screaming"
    response = client.random_openai_command(TARGET_CAPTION, INTERFERER_CAPTION, client)
