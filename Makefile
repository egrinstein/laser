.PHONY:

download-audioset:
	@python data/downloaders/audioset.py

download-audiocaps:
	@python data/downloaders/audiocaps.py

create-audioset-json:
	@python data/json/create_template_json_audioset.py --data_dir ~/datasets/audioset/wavs --output_json config/datafiles/audioset.json

create-audiocaps-json:
	@python data/json/create_template_json_audiocaps.py --wav_dir ~/datasets/audiocaps/ --csv_dir config/datafiles/csvs/ --output_json config/datafiles/

create-audiocaps-csv:
	@python -m data.mixing.audiocaps_csv_mixer --in_csv_dir config/datafiles/csvs --out_csv_dir config/datafiles/csvs

mix-audiocaps:
	@python -m data.mixing.mix_audiocaps --in_csv_dir config/datafiles/csvs/ --out_json_dir config/datafiles/ --in_wav_dir ~/datasets/audiocaps/ --out_wav_dir ~/datasets/audiocaps/mix

train:
	@PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
	@python train.py --workspace workspace/AudioSep --config_yaml config/audiosep_base.yaml --resume_checkpoint_path checkpoint/audiosep_base_4M_steps.ckpt

eval:
	@python -m evaluation.evaluate_audiotext_dataset

submit:
	@qsub qsub.pbs