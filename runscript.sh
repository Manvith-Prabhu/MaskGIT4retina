vit_folder="./pretrained_maskgit/maskgit/"
vqgan_folder="./pretrained_maskgit/vq_vae/"
writer_log="./logs/"
num_worker=4
bsize=32
# Single GPU
python main.py --vit-folder "${vit_folder}" --vqgan-folder "${vqgan_folder}" --writer-log "${writer_log}" --num_workers ${num_worker} --train_config vq_vae
