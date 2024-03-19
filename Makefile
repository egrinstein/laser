.PHONY:

train:
	@PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
	@python train.py --workspace workspace/AudioSep --config_yaml config/audiosep_base.yaml --resume_checkpoint_path checkpoint/audiosep_base_4M_steps.ckpt

dataset:
	@python data/create_template_json_audioset.py --data_dir ~/datasets/audioset/wavs --output_json datafiles/audioset.json

submit:
	@qsub qsub.pbs

benchmark:
	@python benchmark.py --checkpoint_path audiosep_base_4M_steps.ckpt