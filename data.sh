#!/bin/bash
#SBATCH --job-name=data_extract
#SBATCH  --partition=batch_grad
#SBATCH --time=7-0  # 10 hour
#SBATCH --nodelist=ariel-v13
#SBATCH --mem=10G

# hostname
# mkdir /data2/local_datasets/kinetics400
# cd /data2/local_datasets/kinetics400
# tar -xf /data/datasets/test_kinetics400_resized.tar
# tar -xf /data/datasets/train_kinetics400_resized.tar
# tar -xf /data/datasets/val_kinetics400_resized.tar
# cd /local_datasets
# ln -s /data2/local_datasets/kinetics400/

# scp /data/datasets/Kinetics-400.tar.gz.0000 gyeongho@trinity.khu.ac.kr:/data/dataset
# scp /data/datasets/Kinetics-400.tar.gz.0001 gyeongho@trinity.khu.ac.kr:/data/dataset
cd /data2/local_datasets
mkdir Diving48
cd Diving48
tar -xf /data/datasets/tarfiles/Diving48_rgb.tar.gz