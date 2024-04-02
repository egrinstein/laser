import argparse
import os
import json


def add_embeddings_to_audiocaps_json(json_path, embed_dir):
    data = json.load(open(json_path))

    for data_dict in data["data"]:
        embed_path = os.path.join(embed_dir, os.path.basename(data_dict["wav_mixture"]).replace(".wav", ".safetensors"))
        if os.path.exists(embed_path):
            print(f"Embedding added for {data_dict['wav_mixture']}")
            data_dict["command_embedding"] = embed_path

    with open(json_path, 'w') as f:
        json.dump({ "data": data }, f, indent=4)

    print(f"Template json file created at {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a template json file for the audiocaps dataset")
    parser.add_argument("--json_dir", type=str, required=True, help="Path to the directory where the .json files will be updated")
    parser.add_argument("--embed_dir", type=str, required=True, help="Path to the directory containing the embeddings")
    args = parser.parse_args()

    for split in ["train", "val", "test"]:
        embed_dir = os.path.join(args.embed_dir, split)
        json_path = os.path.join(args.json_dir, f"audiocaps_{split}_mix.json")
        print(f"Creating template json file for {split} split")
        add_embeddings_to_audiocaps_json(json_path, embed_dir)
