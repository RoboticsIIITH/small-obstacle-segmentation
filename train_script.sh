CUDA_VISIBLE_DEVICES=0,1 python train.py --resume '/scratch/ash/small_obs/image_geometric_context/exp_0/checkpoint_1_.pth.tar' --ft False --mode train --batch-size 6 --workers 6 --start_epoch 1 --epochs 10 --lr 0.001 --gpu_ids 0,1