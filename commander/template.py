import random
import warnings
import numpy as np
import torch
from optimum.pipelines import pipeline


class TemplateCommandCreator:
    def __init__(self, use_corrector=True):
        self.corrector = CommandCorrector() if use_corrector else None
    
    def __call__(self, caption, interferer_captions, return_type=False):
        return random_template_command(
            caption, interferer_captions,
            corrector=self.corrector, return_type=return_type)


class CommandCorrector:
    def __init__(self):        
        self.model = pipeline(
            "text2text-generation", 
            model='pszemraj/flan-t5-large-grammar-synthesis', 
            accelerator="ort",
            device="cuda",
            # cache_dir='/data/oliveira/tmp'
        )
    
    def __call__(self, text):
        return self.model(text)


POSITIVE_QUERIES = [
    "Keep the {}",
    "Enhance the {}",
    "Amplify the {}",
    "Boost the {}",
    "Increase the {}",
    "Focus on the {}",
    "Extract the {}",
]

NEGATIVE_QUERIES = [
    "Remove the {}",
    "Filter out the {}",
    "Reduce the {}",
    "Eliminate the interfering {}",
    "Discard the {}",
]

MIXED_QUERIES = []

for positive_query in POSITIVE_QUERIES:
    for negative_query in NEGATIVE_QUERIES:
        connectors = [" and ", ", "]
        connector = random.choice(connectors)
        order = np.random.choice([0, 1], size=(1,), p=[1./3, 2./3])
        if order:
            MIXED_QUERIES.append(positive_query + connector + negative_query.lower())
        else:
            MIXED_QUERIES.append(negative_query + connector + positive_query.lower())

DESCRIPTIVE_QUERIES = [
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


def random_template_command(caption: str, interferer_captions: list[str], return_type = False,
                              corrector=None):
    # Sometimes AudioSet's captions have synonyms separated by commas.
    # For example, one class is Accelerating, revving, vroom
    # We can split these and choose one randomly.
        
    def _parse_text(text: str):
        return text.lower()
    
    caption = _parse_text(caption)
    if interferer_captions is not None:
        interferer_captions = [_parse_text(cap) for cap in interferer_captions]
    else:  
        interferer_captions = []

    if len(interferer_captions) == 0:
        Warning("No interference caption provided. Won't use mixed commands")
        command_type = random.choice(COMMAND_TYPES[:-1])
    elif len(interferer_captions) > 1:
        raise ValueError("Only a single interference caption is supported for now.")
    else:
        interferer_captions = interferer_captions[0]
        command_type = random.choice(COMMAND_TYPES)
    
    if command_type == "positive":
        query = random.choice(POSITIVE_QUERIES).format(caption)
    elif command_type == "descriptive":
        query = random.choice(DESCRIPTIVE_QUERIES).format(caption)
    elif command_type == "negative":
        query = random.choice(NEGATIVE_QUERIES).format(interferer_captions)
    elif command_type == "mixed":
        query = random.choice(MIXED_QUERIES).format(caption, interferer_captions)
    else:
        raise ValueError(f"Invalid command type: {command_type}")
    
    if corrector:
        with warnings.catch_warnings(action="ignore"):
            with torch.no_grad():
                query = corrector(query)[0]["generated_text"]

    if return_type:
        return query, command_type
    else:
        return query

if __name__ == "__main__":
    caption = "male speech"
    print(random_template_command(caption))
