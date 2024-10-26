

CUDA_VISIBLE_DEVICES=0,4,7 python scripts/sample_diffusion.py -r /mnt/disk10T/wengtaohan/Code/latent-diffusion-main/logs/2023-04-30T20-01-30_my_lsun/checkpoints/last.ckpt -l sample -n 8 --batch_size 4 -c 500 -e 0