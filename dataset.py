from typing import Optional
import jax
import tensorflow as tf

OPTIONS = tf.data.Options()
OPTIONS.deterministic = True
OPTIONS.autotune.enabled = True

# Feature description for parsing TFRecords
FEATURE_DESCRIPTION = {
    'ids': tf.io.FixedLenFeature([], tf.string, default_value=''),
}

def parse_example(example_proto):
    """Parses a single TFRecord example into a tensor."""
    example = tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)
    return tf.io.decode_raw(example['ids'], tf.uint16)

def get_dataset(pattern: str,
                batch_size: int = 8,
                block_size: int = 1024,
                shuffle_buffer_size: Optional[int] = None,
                repeat: Optional[int] = None,
                seed: Optional[int] = None) -> tf.data.Dataset:

    file_ds = tf.data.Dataset.list_files(pattern, shuffle=bool(shuffle_buffer_size), seed=seed)
    file_ds = file_ds.shard(jax.process_count(), jax.process_index())

    ds = tf.data.TFRecordDataset(file_ds, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    
    if repeat is not None:
        ds = ds.repeat(repeat)

    # Apply shuffling only if a buffer size is specified
    if shuffle_buffer_size:
        ds = ds.shuffle(shuffle_buffer_size)

    # Process tokens into contiguous sequences
    ds = ds.unbatch().batch(block_size + 1, drop_remainder=True)

    # Shuffle again after forming blocks if needed
    if shuffle_buffer_size:
        ds = ds.shuffle(shuffle_buffer_size)

    # Batch for model training
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.batch(jax.local_device_count(), drop_remainder=True)

    ds = ds.with_options(OPTIONS)
    return ds.prefetch(tf.data.AUTOTUNE)
