import tensorflow as tf
count = 0
for tfrecord_file in tf.io.gfile.glob("bridge_subset/bridge_data_v2-train.tfrecord-*"):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    for _ in dataset:
        count += 1
        
print("Episodes:",count)