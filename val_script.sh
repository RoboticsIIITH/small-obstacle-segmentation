CUDA_VISIBLE_DEVICES=0 python train.py --resume '/scratch/ash/small_obs/image_context_temporal/exp_road_prior/checkpoint_6_.pth.tar' --mode val --batch-size 1 --workers 1 --epochs 1 --gpu_ids 0