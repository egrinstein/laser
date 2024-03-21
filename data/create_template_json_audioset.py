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
from tqdm import tqdm

MIN_SIZE_IN_BYTES = 5000


def create_template_json_audioset(data_dir, output_json,
                                  split_train=0.8, split_val=0, split_test=0.2):
    if split_test + split_val + split_train != 1:
        raise ValueError("Split values should sum to 1")
    
    data = []
    for label in tqdm(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for audio_file in os.listdir(label_dir):
            audio_file_path = os.path.join(label_dir, audio_file)
            if os.path.getsize(audio_file_path) < MIN_SIZE_IN_BYTES:
                os.remove(audio_file_path)
                continue
            data.append({
                "wav": audio_file_path,
                "caption": label
            })
    random.shuffle(data)

    # Split data
    data_len = len(data)
    train_data = data[:int(split_train * data_len)]
    val_data = data[int(split_train * data_len):int((split_train + split_val) * data_len)]
    test_data = data[int((split_train + split_val) * data_len):]

    data_dict = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

    for split in data_dict:
        split_data = data_dict[split]
        
        json_data = {"data": split_data}
        with open(output_json.replace(".json", f"_{split}.json"), "w") as f:
            json.dump(json_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a template json file for the AudioSet dataset")
    parser.add_argument("--data_dir", type=str, help="Path to the directory containing the AudioSet dataset")
    parser.add_argument("--output_json", type=str, help="Path to the output json file")
    args = parser.parse_args()
    create_template_json_audioset(args.data_dir, args.output_json)
