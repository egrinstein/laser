# This file is used to create a template json file for the AudioSet dataset.
# A template json has the follwoing structure:
# {
#     "data": [
#      {
#       "wav": "path_to_audio_file",
#       "caption": "textual_desciptions"
#      }
#     ]
# }
# In turn, the AudioSet dataset is a collection of directories, each of which contains a number of audio files.
# The directory name is the label of the audio files in the directory.

import os
import json
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

def create_template_json_audioset(data_dir, output_json):
    data = []
    for label in tqdm(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for audio_file in os.listdir(label_dir):
            audio_file_path = os.path.join(label_dir, audio_file)
            data.append({
                "wav": audio_file_path,
                "caption": label
            })
    random.shuffle(data)
    json_data = {"data": data}
    with open(output_json, "w") as f:
        json.dump(json_data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a template json file for the AudioSet dataset")
    parser.add_argument("--data_dir", type=str, help="Path to the directory containing the AudioSet dataset")
    parser.add_argument("--output_json", type=str, help="Path to the output json file")
    args = parser.parse_args()
    create_template_json_audioset(args.data_dir, args.output_json)