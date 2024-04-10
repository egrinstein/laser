import json
import random

import json

import boto3

boto3_bedrock = boto3.client('bedrock-runtime')

POSITIVE_COMMAND = """
Human: You are an AI assistant working in the field of Language-Queried Audio Sound Separation (LASS).
Given a target caption referring to an audio scene, transform it to a 'separation command'. 
Typical expected verbs in the command are 'Keep', 'Enhance', 'Amplify', 'Boost', 'Increase', 'Focus on' and 'Extract'.
Feel free to simplify the caption to make the command more concise. For example, given a target caption

Target: A woman talks nearby as water pours

Example commands could be:

- Enhance the woman talking and the water pouring
- Keep the woman talking and the water pouring"
- Isolate the woman talking nearby and the water pouring"
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

MODELID = 'anthropic.claude-v2' # change this to use a different version from the model provider

def random_aws_bedrock_command(client, target_caption, interferer_caption, return_type=False):
    command_type = random.choice(COMMAND_TYPES)
    command = COMMANDS[command_type]

    prompt = command
    if command_type == "positive" or command_type == "mixed":
        prompt += f"\n\nHuman:Target: {target_caption}"
    if command_type == "negative" or command_type == "mixed":
        prompt += f"\n\nHuman:Interferer: {interferer_caption}"

    body = json.dumps({
                    "prompt": prompt,
                    "max_tokens_to_sample": 100,
                    "temperature":0.5,
                    "top_k":250,
                    "top_p":0.5,
                    "stop_sequences": ["\n\nHuman:"]
                  }) 
    
    response = client.invoke_model(
        body=body, modelId=MODELID,
        accept='application/json', contentType='application/json')
    
    response_body = json.loads(response.get('body').read())

    response = response_body.get('completion')

    print(response)


class AwsBedrockCommandCreator:
    def __init__(self) -> None:
        self.client = boto3.client(
            service_name='bedrock-runtime', 
            region_name="us-east-1"
        )

    def __call__(self, target_caption, interferer_captions, return_type=False):
        return random_aws_bedrock_command(
            self.client, target_caption, interferer_captions[0],
            return_type=return_type)
        

if __name__ == "__main__":
    client = AwsBedrockCommandCreator()

    TARGET_CAPTION = "A woman talks nearby as water pours"
    INTERFERER_CAPTION = "A man screaming"
    response = client.random_openai_command(TARGET_CAPTION, INTERFERER_CAPTION, client)
