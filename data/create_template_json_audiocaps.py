# This file is used to create a .json file for the audiocaps dataset, which can be loaded as an AudioTextDataset.
# It does so by processing the AudioCaps .csv file.
# The AudioCaps .csv files contain the following columns:
#
# audiocap_id,youtube_id,start_time,caption
# 91139,Zn4ViKWcw,130,A woman talks nearby as water pours
#
#
# A json file is created by iterating over the AudioCaps .csv file and creating a dictionary for each row.
# The json file looks like this:
# {
#     "data": [
#      {
#       "wav": "path/to/[audiocap_id].wav",
#       "caption": [caption]
#      },
#      ... 
#     ]
# }

import os
import json
import pandas as pd
import argparse

def create_template_json_audiocaps(csv_file, audiocaps_wav_path, output_json):
    # Read the csv file
    df = pd.read_csv(csv_file)
    data = []

    for index, row in df.iterrows():
        # Create the dictionary
        data_dict = {
            "wav": os.path.join(
                audiocaps_wav_path, f"{row['audiocap_id']}.wav"),
            "caption": row['caption']
        }
        data.append(data_dict)

    # Write the dictionary to a json file
    with open(output_json, 'w') as f:
        json.dump({ "data": data}, f, indent=4)
    
    print(f"Template json file created at {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a template json file for the audiocaps dataset")
    parser.add_argument("--csv_dir", type=str, required=True, help="Path to the directory containing the files train.csv, val.csv, test.csv")
    parser.add_argument("--wav_dir", type=str, required=True, help="Path to the audiocaps wav files")
    parser.add_argument("--output_json", type=str, required=True, help="Path to the directory where the .json files will be saved")
    args = parser.parse_args()
    
    for split in ["train", "val", "test"]:
        csv_file = os.path.join(args.csv_dir, f"{split}.csv")
        output_json = os.path.join(args.output_json, f"audiocaps_{split}.json")
        print(f"Creating template json file for {split} split")
        create_template_json_audiocaps(csv_file, args.wav_dir, output_json)