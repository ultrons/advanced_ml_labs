#!/usr/bin/env python

"""
Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


"""


import argparse
import json
import os

import model

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_bucket',
        help = 'GCS location where the \'input\' directory containing the dataset files are found',
        required = True
    )

    parser.add_argument(
        '--output_dir',
        help = 'GCS location to write checkpoints and export models',
        required = True
    )
    parser.add_argument(
        '--trainfile',
        help = 'File to use for training',
        default = 'train.csv'
    )
    parser.add_argument(
        '--evalfile',
        help = 'File to use for evaluation',
        default = 'eval.csv'
    )
    parser.add_argument(
        '--estimator_type',
        help = 'Type of estimator to use. Supported: wd, linear',
        default = 'wd'
    )
    parser.add_argument(
        '--train_examples',
        help = 'Number of examples (in thousands) to run the training job over. If this is more than actual # of examples available, it cycles through them. So specifying 1000 here when you have only 100k examples makes this 10 epochs.',
        type = int,
        default = 2000
    )
    parser.add_argument(
        '--batch_size',
        help = 'Number of examples to compute gradient over.',
        type = int,
        default = 512
    )
    parser.add_argument(
        '--nlayers',
        help = 'Number of layers in the deep part of the neural network',
        type = int,
        default = 3
    )
    parser.add_argument(
        '--first_layer_size',
        help = 'Number of units in first layer of the deep part of the neural network',
        type = int,
        default = 128
    )
    parser.add_argument(
        '--layer_scale_factor',
        help = 'Factor by which to scale subsequent layer sizes in the deep part of the neural network',
        type = float,
        default = 0.5
    )
    parser.add_argument(
        '--activation',
        help = 'Activation function to use',
        default = tf.nn.relu
    )
    parser.add_argument(
        '--dropout',
        help = 'With probability p (the input number), outputs the elements scaled up by 1 / p so the expected sum is unchanged, otherwise outputs 0',
        type = float,
        default = 0.001
    )
    parser.add_argument(
        '--dnn_optimizer',
        help = 'Optimizer used. Supported names: Adagrad, Adam, Ftrl, RMSProp, SGD (all case sensitive)',
        default = 'ProximalAdagrad'
    )
    parser.add_argument(
        '--DNN_LR',
        help = 'The learning rate used by the DNN optimizer in a wide and deep network',
        type = float,
        default = 0.01
    )
    parser.add_argument(
        '--DNN_L1',
        help = 'The L1 regularization strength used by the DNN optimizer in a wide and deep network',
        type = float,
        default = 0.001
    )
    parser.add_argument(
        '--DNN_L2',
        help = 'The L1 regularization strength used by the DNN optimizer in a wide and deep network',
        type = float,
        default = 0.001
    )
    parser.add_argument(
        '--linear_optimizer',
        help = 'Optimizer used. Supported names: Adagrad, Adam, Ftrl, RMSProp, SGD (all case sensitive)',
        default = 'Ftrl'
    )
    parser.add_argument(
        '--LIN_LR',
        help = 'The learning rate used by the linear optimizer in a wide and deep network',
        type = float,
        default = 0.01
    )
    parser.add_argument(
        '--LIN_LR_POWER',
        help = 'The learning rate power used by the linear optimizer in a wide and deep network. Must be less than or equal to zero',
        type = float,
        default = -0.5
    )
    parser.add_argument(
        '--LIN_L1',
        help = 'The L1 regularization strength used by the linear optimizer in a wide and deep network',
        type = float,
        default = 0.001
    )
    parser.add_argument(
        '--LIN_L2',
        help = 'The L1 regularization strength used by the linear optimizer in a wide and deep network',
        type = float,
        default = 0.001
    )
    parser.add_argument(
        '--LIN_LOCKING',
        help = 'Whether locks are used for update operations by the linear optimizer in a wide and deep network',
        default = False
    )
    parser.add_argument(
        '--LIN_SHRINKAGE',
        help = 'This is a magnitude penalty (LIN_L2 is a stabilization penalty) and only occurs on active weights',
        type = float,
        default = 0.001
    )
    parser.add_argument(
        '--job-dir',
        help = 'this model ignores this field, but it is required by gcloud',
        default = 'junk'
    )
    parser.add_argument(
        '--eval_steps',
        help = 'Positive number of steps for which to evaluate model. Default to None, which means to evaluate until input_fn raises an end-of-input exception',
        type = int,
        default = None
    )

    args = parser.parse_args()
    arguments = args.__dict__

    output_dir = arguments.pop('output_dir')
#     model.TRAINFILE = arguments.pop('trainfile')
#     model.EVALFILE = arguments.pop('evalfile')
    model.INPUT_BUCKET = arguments.pop('input_bucket')
    model.BATCH_SIZE = arguments.pop('batch_size')
    model.EVAL_STEPS = arguments.pop('eval_steps')
    model.TRAIN_STEPS = (arguments.pop('train_examples') * 1000) / model.BATCH_SIZE
    print ("Will train for {} steps using batch_size={}".format(model.TRAIN_STEPS, model.BATCH_SIZE))
    model.ESTIMATOR_TYPE = arguments.pop('estimator_type')
    model.FIRST_LAYER_SIZE = arguments.pop('first_layer_size')
    model.LAYER_SCALE_FACTOR = arguments.pop('layer_scale_factor')
    model.NLAYERS = arguments.pop('nlayers')
    model.HIDDEN_UNITS = [max(2, int(model.FIRST_LAYER_SIZE * model.LAYER_SCALE_FACTOR**i)) \
                          for i in range(model.NLAYERS)]
    model.DROPOUT = arguments.pop('dropout')
    model.DNN_OPTIMIZER = arguments.pop('dnn_optimizer')
    model.DNN_LR = arguments.pop('DNN_LR')
    model.DNN_L1 = arguments.pop('DNN_L1')
    model.DNN_L2 = arguments.pop('DNN_L2')
    model.LIN_OPTIMIZER = arguments.pop('linear_optimizer')
    model.LIN_LR = arguments.pop('LIN_LR')
    model.LIN_LR_POWER = arguments.pop('LIN_LR_POWER')
    model.LIN_L1 = arguments.pop('LIN_L1')
    model.LIN_L2 = arguments.pop('LIN_L2')
    model.LIN_LOCKING = arguments.pop('LIN_LOCKING')
    model.LIN_SHRINKAGE = arguments.pop('LIN_SHRINKAGE')

    # Append trial_id to path
    output_dir = os.path.join(
        output_dir,
        json.loads(
            os.environ.get('TF_CONFIG', '{}')
        ).get('task', {}).get('trial', '')
    )

    # Run the training job
    model.train_and_evaluate(output_dir)
