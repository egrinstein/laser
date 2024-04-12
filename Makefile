.PHONY:

download:
	@python data/download_audiocaps.py --out_wav_dir ~/datasets/audiocaps --out_csv_dir config/datafiles/csvs --n_jobs 5
	
csv:
	@python -m data.mixing.audiocaps_csv_mixer --in_csv_dir config/datafiles/csvs --out_csv_dir config/datafiles/csvs

mix:
	@python -m data.mixing.audiocaps_wav_mixer --in_csv_dir config/datafiles/csvs/ --out_json_dir config/datafiles/ --in_wav_dir ~/datasets/audiocaps/ --out_wav_dir ~/datasets/audiocaps/mix

embeddings:
	@python -m data.create_audiocaps_embeddings --in_dir config/datafiles --out_dir ~/datasets/audiocaps/command_embeddings --mode e2e --n_jobs 5

add-embeddings-to-json:
	@python data/add_embeddings_to_audiocaps_json.py --embed_dir ~/datasets/audiocaps/bert_embeddings_commands --json_dir config/datafiles/

train:
	@PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
	@python train.py --workspace workspace/AudioSep --config_yaml config.yaml

eval:
	@python -m evaluation.evaluate_audiotext_dataset

submit:
	@qsub qsub.pbs