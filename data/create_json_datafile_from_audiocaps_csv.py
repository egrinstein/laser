# This file processes the .csv file from AudioCaps and creates a JSON file with the following structure:
# A template json has the follwoing structure:
# {
#     "data": [
#      {
#       "wav": "path_to_audio_file",
#       "caption": "textual_desciptions"
#      }
#     ]
# }
# In turn, the AudioCaps .csv has the following structure:
# ,audiocap_id,youtube_id,start_time,caption,# YTID,start_seconds,end_seconds,positive_labels,audioset_caption
# 0,91139,r1nicOVtvkQ,130,A woman talks nearby as water pours,r1nicOVtvkQ,130.0,140.0,"/m/02jz0l,/m/09x0r","Water tap, faucet| Speech"

import json
import os
import pandas as pd


def create_json_datafile_from_audiocaps_csv(csv_path: str,
                                            audiocaps_dataset_path: str,
                                            out_json_path: str):
    df = pd.read_csv(csv_path)
    data = []
    for _, row in df.iterrows():
        wav_path = os.path.join(audiocaps_dataset_path, str(row["audiocap_id"]) + ".wav")
        if not os.path.exists(wav_path):
            continue
        else:
            data.append({
                "wav": wav_path,
                "caption": row["audioset_caption"]
            })
    
    with open(out_json_path, "w") as f:
        json.dump({"data": data}, f, indent=4)


if __name__ == "__main__":
    create_json_datafile_from_audiocaps_csv(
        csv_path="config/datafiles/csvs/train_audiocaps.csv",
        audiocaps_dataset_path="/Users/ezajlerg/datasets/audiocaps/train",
        out_json_path="config/datafiles/train_audiocaps.json"
    )
