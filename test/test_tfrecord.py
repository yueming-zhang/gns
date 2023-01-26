import tensorflow as tf
import functools
import os
import json
from pathlib import Path
import numpy as np
import sys
import pickle

sys.path.insert(0, str(Path(__file__).parents[1]))
from gns import reading_utils_tf

# Read the data back out.

def decode_fn(record_bytes):
    return tf.io.parse_single_example(
        # Data
        record_bytes,

        # Schema
        {"x": tf.io.FixedLenFeature([], dtype=tf.float32),
        "y": tf.io.FixedLenFeature([], dtype=tf.float32)}
    )

def test_read_tfrecrod():
    data_path = "/home/ming/dev/learning_to_simulate/datasets/WaterRamps"
    metadata = _read_metadata(data_path)    
    ds = tf.data.TFRecordDataset([os.path.join(data_path, f'train.tfrecord')])

    ds = ds.map(functools.partial(reading_utils_tf.parse_serialized_simulation_example, metadata=metadata))

    i = 0
    for batch in ds:
        # print(batch[0]['particle_type'].shape)
        # print(batch[1]['position'].shape)
        print (f'{i} - {batch[1]["position"].shape}')
        i+=1

    pass

def test_save_tfrecrod_to_npz():
    data_type = 'valid'
    data_path = "/mnt/d/Tmp/MultiMaterial"
    # np_data_path = f"/home/ming/dev/gns/data/Water-3D/{data_type}.npz"
    np_data_path = f"/mnt/d/Tmp/MultiMaterial/{data_type}.npz"

    np_list = []
    metadata = _read_metadata(data_path)    
    ds = tf.data.TFRecordDataset([os.path.join(data_path, f'{data_type}.tfrecord')])

    ds = ds.map(functools.partial(reading_utils_tf.parse_serialized_simulation_example, metadata=metadata))
    ds_list = []
    for idx, batch in enumerate(ds):
        particle_type = batch[0]['particle_type'].numpy()
        position = batch[1]['position'].numpy()
        ds_list.append((f'{idx}', [position, particle_type]))
    pass
    pickle.dump(ds_list, open(np_data_path, 'wb'))

def _read_metadata(data_path):
  with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
    return json.loads(fp.read())
