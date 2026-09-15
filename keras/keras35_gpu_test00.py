import tensorflow as tf
print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

if(gpus):
    try:
        print(
            'gpu 있다.'
        )
    except RuntimeError as e:
        print(e)