.PHONY:

download-audioset:
	@python data/download_audioset.py

dataset-json:
	@python data/create_template_json_audioset.py --data_dir ~/datasets/audioset/wavs --output_json config/datafiles/audioset.json

train:
	@PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
	@python train.py --workspace workspace/AudioSep --config_yaml config/audiosep_base.yaml --resume_checkpoint_path checkpoint/audiosep_base_4M_steps.ckpt

eval:
	@python -m evaluation.evaluate_audiotext_dataset

submit:
	@qsub qsub.pbs