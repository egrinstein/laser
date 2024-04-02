from openai import OpenAI

COMMAND = """
Given a target and an interferer captions referring to an audio scene,
create a json with a key 'result',
whose value contains a list of {} separation command options. A separation is itself a json containing
two keys, 'command' and 'type'. A separation command be one of 3 types, 'positive', 'negative' or 'mixed'.
For example:

Target: A woman talks nearby as water pours
Interferer: Dishes are clanging

Positive command example: Enhance the sound of the woman talking nearby and the water pouring
Negative command example: Remove the clanging dishes from the audio
Mixed command example: Keep the sound of the woman talking nearby and the water pouring, and exclude the clanging dishes from the mix
"""

def random_openai_command(client, target_caption, interferer_caption, num_commands=5):
    return client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are an AI assistant working in the field of Language-Queried Audio Sound Separation (LASS)."},
        {"role": "user", "content": COMMAND.format(num_commands)},
        {"role": "user", "content": f"Target: {target_caption}"},
        {"role": "user", "content": f"Interferer: {interferer_caption}"},
    ],
    response_format = { "type": "json_object" }
    ).choices[0].message.content


class OpenAiCommandCreator:
    def __init__(self) -> None:
        self.client = OpenAI()

    def __call__(self, target_caption, interferer_captions, num_commands=5):
        return random_openai_command(
            self.client, target_caption, interferer_captions[0], num_commands)


if __name__ == "__main__":
    client = OpenAiCommandCreator()

    TARGET_CAPTION = "A woman talks nearby as water pours"
    INTERFERER_CAPTION = "A man screaming"
    response = client.random_openai_command(TARGET_CAPTION, INTERFERER_CAPTION, client)
