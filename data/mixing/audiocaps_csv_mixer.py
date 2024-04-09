# This file is used to mix the rows on the AudioCaps dataset csv file for the purpose of training and testing
# an audio separation model.
# The AudioCaps csv has the following columns:
# ,audiocap_id,youtube_id,start_time,caption,audioset_caption
#0,91139,r1nicOVtvkQ,130,A woman talks nearby as water pours,"/m/02jz0l,/m/09x0r"
#1,58146,UDGBjjwyaqE,20,Multiple clanging and clanking sounds,
# The mixer will mix the rows in the csv file to create a new csv file with the following columns:
# ,audiocap_id_target,audiocap_id_interferer, youtube_id_target, youtube_id_interferer, start_time_target, start_time_interferer, caption_target, caption_interferer, caption_similarity

import argparse
import os
import pandas as pd
import random
import torch

from tqdm import trange

allowed_labels = [
    '/m/0395lw', '/m/028ght', '/m/0bt9lr',
    '/m/02qldy', '/m/01jwx6', '/m/03kmc9', '/m/09ddx',
    '/m/07r_k2n', '/m/0ngt1', '/m/07qdb04', '/m/02zsn',
    '/m/0jbk', '/m/03j1ly', '/m/0dbvp', '/m/0jb2l',
    '/m/015p6', '/m/068hy', '/m/09x0r', '/m/0316dw',
    '/m/012ndj', '/m/01m2v', '/m/07rc7d9', '/m/06mb1',
    '/m/012n7d', '/m/01j3sz', '/t/dd00038', '/m/05zppz',
    '/m/04qvtq', '/m/07r10fb', '/t/dd00001'
]


def mix_audiocaps_csv(input_csv, output_csv, n_mix=-1,
                       allowed_indices_csv=None):
    df = pd.read_csv(input_csv)
    positive_labels = set()

    # Filter rows that are not in the allowed indices csv
    if allowed_indices_csv:
        n_total = len(df)
        allowed_indices = pd.read_csv(allowed_indices_csv, header=None)
        df = df[df['youtube_id'].isin(allowed_indices[0].values)]
        print(f"Filtered {n_total - len(df)} rows from the csv file")
    # Filter rows that do not have any of the allowed labels
    n_total = len(df)
    df = df[df['positive_labels'].apply(lambda x: any(label in x for label in allowed_labels))]
    print(f"Filtered {n_total - len(df)} rows without labels from the csv file")

    n = len(df)

    if n_mix == -1:
        n_mix = n

    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    # create a new dataframe
    mixed_df = []

    for i in trange(n_mix):
        target_row = df.iloc[i%n]


        interferer_row = df.iloc[random.randint(0, n-1)]

        n_attempts = 0
        labels_target = set(target_row['positive_labels'].split(','))
        labels_interferer = set(interferer_row['positive_labels'].split(','))
        while _shared_captions(labels_target, labels_interferer):
            n_attempts += 1
            interferer_row = df.iloc[random.randint(0, n-1)]
            labels_interferer = set(interferer_row['positive_labels'].split(','))
            #print(f"Attempt {n_attempts} to find a non-shared caption")
        
        positive_labels.update(labels_target)
        positive_labels.update(labels_interferer)

        mixed_row = {
            'audiocap_id_target': target_row['audiocap_id'],
            'audiocap_id_interferer': interferer_row['audiocap_id'],
            'youtube_id_target': target_row['youtube_id'],
            'youtube_id_interferer': interferer_row['youtube_id'],
            'start_time_target': target_row['start_time'],
            'start_time_interferer': interferer_row['start_time'],
            'caption_target': target_row['caption'],
            'caption_interferer': interferer_row['caption'],
        }
        # print(mixed_row)
        mixed_df.append(mixed_row)

    mixed_df = pd.DataFrame(mixed_df)
    mixed_df.to_csv(output_csv, index=False)


def _shared_captions(target_captions: set, interferer_captions: set):
    return len(target_captions.intersection(interferer_captions)) > 0

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Create a template json file for the audiocaps dataset")
    parser.add_argument("--in_csv_dir", type=str, required=True, help="Path to the directory containing the files train.csv, val.csv, test.csv")
    parser.add_argument("--out_csv_dir", type=str, required=True, help="Path to the directory where the .json files will be saved")
    args = parser.parse_args()
    
    n_splits = [40000, 4000, 8000]
    for i,split in enumerate(["train", "val", "test"]):
        in_csv_file = os.path.join(args.in_csv_dir, f"{split}_audiocaps.csv")
        output_csv = os.path.join(args.out_csv_dir, f"audiocaps_{split}_mix.csv")
        
        if split == "train":
            allowed_indices_csv = os.path.join(args.in_csv_dir, f"lassnet_training_indexes.csv")
        else:
            allowed_indices_csv = None

        print(f"Creating mix.csv file for {split} split")
        
        mix_audiocaps_csv(in_csv_file, output_csv, n_mix=n_splits[i], allowed_indices_csv=allowed_indices_csv)
