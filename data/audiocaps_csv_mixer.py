# This file is used to mix the rows on the AudioCaps dataset csv file for the purpose of training and testing
# an audio separation model.
# The AudioCaps csv has the following columns:
# ,audiocap_id,youtube_id,start_time,caption
#0,91139,r1nicOVtvkQ,130,A woman talks nearby as water pours,
#1,58146,UDGBjjwyaqE,20,Multiple clanging and clanking sounds,
# The mixer will mix the rows in the csv file to create a new csv file with the following columns:
# ,audiocap_id_target,audiocap_id_interferer, youtube_id_target, youtube_id_interferer, start_time_target, start_time_interferer, caption_target, caption_interferer, caption_similarity

import os
import pandas as pd
import torch

from tqdm import trange

from models.clap_encoder import CLAP_Encoder


class ClapSimilarity:
    def __init__(self) -> None:
        self.clap_encoder = CLAP_Encoder().eval()
        
    def __call__(self, query1, query2):
        query_embeddings = self.clap_encoder(
            modality="text", text=[query1, query2],
        )
        distance = torch.nn.functional.cosine_similarity(
            query_embeddings[0].unsqueeze(0),
            query_embeddings[1].unsqueeze(0)
        )
        return distance.item()


def mix_audiocaps_csv(input_csv, output_csv, n_mix=-1, caption_similarity_func=None,
                       caption_similarity_threshold=0.2):
    df = pd.read_csv(input_csv)
    
    if n_mix == -1:
        n_mix = len(df)

    if caption_similarity_func is None:
        caption_similarity_func = lambda x, y: 1

    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    # create a new dataframe
    mixed_df = []

    for i in trange(n_mix):
        target_row = df.iloc[i%n_mix]
        interferer_row = df.iloc[(i+1) % n_mix]

        caption_similarity = caption_similarity_func(
            target_row['caption'], interferer_row['caption'])
        
        n_attempts = 0
        while caption_similarity > caption_similarity_threshold :
            print(f"Skipping pair with caption distance {caption_similarity} Target: {target_row['caption']}, Interferer: {interferer_row['caption']}")
            n_attempts += 1
            interferer_row = df.iloc[(i+1+n_attempts) % n_mix]
            caption_similarity = caption_similarity_func(
                target_row['caption'], interferer_row['caption'])
        mixed_row = {
            'audiocap_id_target': target_row['audiocap_id'],
            'audiocap_id_interferer': interferer_row['audiocap_id'],
            'youtube_id_target': target_row['youtube_id'],
            'youtube_id_interferer': interferer_row['youtube_id'],
            'start_time_target': target_row['start_time'],
            'start_time_interferer': interferer_row['start_time'],
            'caption_target': target_row['caption'],
            'caption_interferer': interferer_row['caption'],
            'caption_similarity': caption_similarity
        }
        # print(mixed_row)
        mixed_df.append(mixed_row)

    mixed_df = pd.DataFrame(mixed_df)
    mixed_df.to_csv(output_csv, index=False)


if __name__ == '__main__':
    input_csv = 'config/datafiles/csvs/train_audiocaps.csv'
    output_csv = 'config/datafiles/csvs/train_audiocaps_mixed.csv'

    print("Loading the query encoder (CLAP)...")
    clap_similarity = ClapSimilarity()
    mix_audiocaps_csv(input_csv, output_csv, n_mix=1000, caption_similarity_func=clap_similarity)