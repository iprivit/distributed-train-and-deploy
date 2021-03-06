import tensorflow as tf
import argparse
import os
import json
import subprocess
import numpy as np


def model(x_train, y_train, x_test, y_test,  epochs=8, batch_size=512, base_learning_rate=0.0002, target_shape=20):
    # the following line allows us to train with multiple GPUs using data parallelism
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=None, include_top=False, weights='imagenet', input_tensor=None, pooling=None, classes=1000) # mobilenet_v2.MobileNetV2
        base_model.trainable = True
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        #addl_dense = tf.keras.layers.Dense(256,activation='relu')
        prediction_layer = tf.keras.layers.Dense(target_shape, activation='softmax')
        model = tf.keras.Sequential([
                      base_model,
                      global_average_layer,
                      #addl_dense,
                      prediction_layer
            ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                              metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=epochs, batch_size=batch_size)
    #model.evaluate(x_test, y_test)
    return model


def _load_training_data(base_dir):
    """Load training data"""
    x_train = np.load(os.path.join(base_dir, 'train_data.npy'))
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.load(os.path.join(base_dir, 'train_labels.npy'))
    y_train = np.array(y_train, dtype=np.float32)
    return x_train, y_train


def _load_testing_data(base_dir):
    """Load testing data"""
    x_test = np.load(os.path.join(base_dir, 'eval_data.npy'))
    x_test = np.array(x_test, dtype=np.float32)
    y_test = np.load(os.path.join(base_dir, 'eval_labels.npy'))
    y_test = np.array(y_test, dtype=np.float32)
    return x_test, y_test


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--target-shape', type=int, default=20)

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()

    train_data, train_labels = _load_training_data(args.train)
    eval_data, eval_labels = _load_testing_data(args.train)
    
    mobile_classifier = model(train_data,train_labels,eval_data,eval_labels, epochs=args.epochs, batch_size=args.batch_size,
                             base_learning_rate=args.lr, target_shape=args.target_shape)

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001'
        mobile_classifier.save(os.path.join(args.sm_model_dir, '000000001'), 'mobilenet_model.h5')


        
        
        
