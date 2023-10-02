
cd /data/dataset/ActivityNet-full

echo "train tar files count:"
tar -tf v1-2_train.tar.gz | wc -l

echo "val tar files count:"
tar -tf v1-2_val.tar.gz | wc -l

echo "train_val tar files count:"
tar -tf v1-3_train_val.tar.gz | wc -l