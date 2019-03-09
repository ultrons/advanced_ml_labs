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

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

tf.logging.set_verbosity(tf.logging.INFO)

N_CLASSES = 4

# General hyperparameters
ESTIMATOR_TYPE = 'wd'
TRAIN_STEPS = 10001
EVAL_STEPS = None
BATCH_SIZE = 512
EVAL_INTERVAL = 30

# Hyperparameters to use for a neural network
NLAYERS = 4
FIRST_LAYER_SIZE = 128
LAYER_SCALE_FACTOR = 0.5
HIDDEN_UNITS = [max(2, int(FIRST_LAYER_SIZE * LAYER_SCALE_FACTOR**i)) for i in range(NLAYERS)]
ACTIVATION = tf.nn.relu
DROPOUT = 0.001
DNN_OPTIMIZER = 'ProximalAdagrad'
LIN_OPTIMIZER = 'Ftrl'
# Parameters for the DNN optimizer (Adagrad)
DNN_LR = 0.01
DNN_L1 = 0.001
DNN_L2 = 0.001
# Parameters for the linear optimizer (Ftrl)
LIN_LR = 0.01
LIN_LR_POWER = -0.5
LIN_L1 = 0.0
LIN_L2 = 0.0
LIN_LOCKING = False
LIN_SHRINKAGE = 0.0

KEY_COLUMN = 'loan_sequence_number'
LABEL_COLUMN = 'TARGET'
INPUT_BUCKET='loan-delinq-bucket'
# Specify train file name (and path, if necessary)
TRAINFILE = 'train.csv'

# Specify eval file name (and path, if necessary)
EVALFILE = 'eval.csv'



bool_cols = []
int_cols = ['credit_score', 'mortgage_insurance_percentage', 'Number_of_units', 'cltv', 'original_upb',
            'ltv', 'original_loan_term', 'number_of_borrowers','min_CURRENT_DEFERRED_UPB']
str_cols = ['first_time_home_buyer_flag', 'occupancy_status', 'channel', 'property_state',
            'property_type', 'loan_purpose', 'seller_name', 'service_name']
str_nuniques = [2, 3, 3, 52, 5, 2, 20, 24]
float_cols = ['metropolitan_division', 'original_interest_rate', 'min_CURRENT_ACTUAL_UPB', 'max_CURRENT_ACTUAL_UPB',
              'Range_CURRENT_ACTUAL_UPB', 'stdev_CURRENT_ACTUAL_UPB', 'mode_CURRENT_ACTUAL_UPB', 'average_CURRENT_ACTUAL_UPB',
              'max_CURRENT_DEFERRED_UPB', 'Range_CURRENT_DEFERRED_UPB', 'mode_CURRENT_DEFERRED_UPB', 'average_CURRENT_DEFERRED_UPB',
              'stdev_CURRENT_DEFERRED_UPB', 'min_CURRENT_INTEREST_RATE', 'max_CURRENT_INTEREST_RATE', 'Range_CURRENT_INTEREST_RATE',
              'mode_CURRENT_INTEREST_RATE', 'stdev_CURRENT_INTEREST_RATE', 'average_CURRENT_INTEREST_RATE',
              'PREFINAL_LOAN_DELINQUENCY_STATUS', 'frequency_0', 'frequency_1', 'frequency_2', 'frequency_3',
              'Recency_0', 'Recency_1', 'Recency_2', 'Recency_3']
DEFAULTS = [[''] for col in bool_cols] + [[0] for col in int_cols] + [[0.0] for col in float_cols] + \
           [[''] for col in str_cols] + [[''],[0]]
CSV_COLUMNS = bool_cols + int_cols + float_cols + str_cols + [KEY_COLUMN,LABEL_COLUMN]


def serving_input_fn():
    feature_placeholders = {}

    if len(bool_cols) > 0:
        for col in bool_cols:
            feature_placeholders[col] = tf.placeholder(tf.string, [None])

    if len(int_cols) > 0:
        for col in int_cols:
            feature_placeholders[col] = tf.placeholder(tf.int64, [None])

    if len(float_cols) > 0:
        for col in float_cols:
            feature_placeholders[col] = tf.placeholder(tf.float64, [None])

    if len(str_cols) > 0:
        for col in str_cols:
            feature_placeholders[col] = tf.placeholder(tf.string, [None])

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }

    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

def read_dataset(filename, mode, batch_size=BATCH_SIZE):
    def _input_fn():
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            features.pop(KEY_COLUMN)
            label = features.pop(LABEL_COLUMN)
            return features, label

        # create file path
        file_path = '{}/output/{}'.format(INPUT_BUCKET, filename)

        # Create list of files that match pattern (we are currently not using a pattern
        #   such as 1-of-15)
        file_list = tf.gfile.Glob(file_path)

        # Create dataset from file list
        dataset = (tf.data.TextLineDataset(file_list)  # Read text file
                    .map(decode_csv))  # Transform each elem by applying decode_csv fn

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
            dataset = dataset.shuffle(buffer_size = 10 * batch_size)
        else:
            num_epochs = 1 # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()
    return _input_fn

def get_wd_columns():
    # Define column types. We may also want to construe some columns
    # as categorical as with: age_buckets = tf.feature_column.bucketized_column(
    #                                       age, boundaries=[18, 25, 30, 40, 50, 60, 65])
    # This approach does not allow for bucketization of any cols, regards all int and float
    #  cols as deep while all str and bool cols as wide

    # Continuous columns are deep and have a complex relationship with the output
    deep_columns = []
    if len(float_cols) > 0:
        deep_columns += [tf.feature_column.numeric_column(col) for col in float_cols]
    if len(int_cols) > 0:
        deep_columns += [tf.feature_column.numeric_column(col) for col in int_cols]

    # Our base columns will be the ones that seem to hold categorical information
    base_columns = []
    if len(str_cols) > 0:
        base_columns += [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket(
            col, hash_bucket_size = max(10+num, int(2 * num)))) for col, num in zip(str_cols, str_nuniques)]
    if len(bool_cols) > 0:
        base_columns += [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket(
            col, hash_bucket_size = 3)) for col in bool_cols]

    # Sparse columns are wide and have a linear relationship with the output
    wide_columns = base_columns

    return wide_columns, deep_columns

# forward to key-column to export
def forward_key_to_export(estimator):
    estimator = tf.contrib.estimator.forward_features(estimator)
    # return estimator

    config = estimator.config
    def model_fn2(features, labels, mode):
      estimatorSpec = estimator._call_model_fn(features, labels, mode, config=config)
      if estimatorSpec.export_outputs:
        for ekey in ['predict', 'serving_default']:
          if (ekey in estimatorSpec.export_outputs and
              isinstance(estimatorSpec.export_outputs[ekey],
                         tf.estimator.export.PredictOutput)):
               estimatorSpec.export_outputs[ekey] = \
                 tf.estimator.export.PredictOutput(estimatorSpec.predictions)
      return estimatorSpec
    return tf.estimator.Estimator(model_fn=model_fn2, config=config)
    ##

# create metric for hyperparameter tuning
def my_metrics(labels, predictions):
    pred_values = predictions['class_ids']
    p, r = tf.metrics.precision(labels, pred_values), tf.metrics.recall(labels, pred_values)
    return { \
              'precision'       : p,
              'recall'          : r,
              'mpcAccuracy'     : tf.metrics.mean_per_class_accuracy(labels, tf.reshape(pred_values, [-1]), num_classes=N_CLASSES)
            }

def train_and_evaluate(output_dir):
    run_config = tf.estimator.RunConfig(save_checkpoints_secs = EVAL_INTERVAL,
                                        keep_checkpoint_max = 3)
    wide_columns, deep_columns = get_wd_columns()
    if ESTIMATOR_TYPE == 'wd':
        estimator = tf.estimator.DNNLinearCombinedClassifier(
            model_dir = output_dir,
            n_classes = N_CLASSES,
            linear_optimizer=tf.train.FtrlOptimizer(
                learning_rate=LIN_LR,
                learning_rate_power=LIN_LR_POWER,
                l1_regularization_strength=LIN_L1,
                l2_regularization_strength=LIN_L2,
                use_locking=LIN_LOCKING,
                l2_shrinkage_regularization_strength=LIN_SHRINKAGE
            ),
            dnn_optimizer=tf.train.AdagradOptimizer(
                learning_rate=DNN_LR
            ),
            linear_feature_columns = wide_columns,
            dnn_feature_columns = deep_columns,
            dnn_activation_fn=ACTIVATION,
            dnn_dropout=DROPOUT,
            dnn_hidden_units = HIDDEN_UNITS,
            config=run_config
        )
    else: # ESTIMATOR_TYPE == 'linear' catchall is to use a linear model in case e_spec != 'wd'
      print('>>> using linear')
      estimator = tf.estimator.LinearClassifier(
            model_dir=output_dir,
            feature_columns=wide_columns,
            optimizer=tf.train.FtrlOptimizer(
                learning_rate=LIN_LR,
                l2_regularization_strength=LIN_L2
            ),
            config=run_config
      )

    estimator = tf.contrib.estimator.add_metrics(estimator, my_metrics)
    estimator = forward_key_to_export(estimator)

    train_spec = tf.estimator.TrainSpec(
        input_fn = read_dataset(TRAINFILE,
                                mode = tf.estimator.ModeKeys.TRAIN,
                                batch_size = BATCH_SIZE),
        max_steps = TRAIN_STEPS)

    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn = read_dataset(EVALFILE,
                                mode = tf.estimator.ModeKeys.EVAL,
                                batch_size = 2**15), # don't need to batch in eval
        steps = EVAL_STEPS,
        start_delay_secs = 60, # start evaluating after N seconds
        throttle_secs = 60,  # evaluate every N seconds
        exporters = exporter)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
