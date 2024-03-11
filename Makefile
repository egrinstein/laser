.PHONY:

train:
	@python train.py --workspace workspace/AudioSep --config_yaml config/audiosep_base.yaml --resume_checkpoint_path checkpoint/audiosep_base_4M_steps.ckpt
