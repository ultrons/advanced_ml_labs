#!/bin/bash

# Copyright 2019 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



BUCKET=$1
if [ "$#" -lt 1 ]; then
    echo "Usage Error!"
    exit
fi
set -e
while [ $# -ne 0 ]; do
    case "$1" in
       -h|--help)      echo "Usage: ./train.sh --bucket <bucket-name> --hyperjob <job_id>"
                        exit
                        shift
                        ;;
       -b|--bucket)     BUCKET=$2
                        shift
                        ;;
       -h|--hyperjob)     HYPERJOB=$2
                        shift
                        ;;
       *)               shift
                        ;;
    esac
done   
     

#HYPERJOB=wd_hcr_hptuning_190304_0220 #TODO: Make it an input
TFVERSION=1.8
REGION=us-central1

echo "Extracting information for job $HYPERJOB"

# get information from the best hyperparameter job
DNN_LR=$(gcloud ml-engine jobs describe $HYPERJOB --format 'value(trainingOutput.trials.hyperparameters.DNN_LR.slice(0))')
LIN_L1=$(gcloud ml-engine jobs describe $HYPERJOB --format 'value(trainingOutput.trials.hyperparameters.LIN_L1.slice(0))')
LIN_L2=$(gcloud ml-engine jobs describe $HYPERJOB --format 'value(trainingOutput.trials.hyperparameters.LIN_L2.slice(0))')
LIN_LR=$(gcloud ml-engine jobs describe $HYPERJOB --format 'value(trainingOutput.trials.hyperparameters.LIN_LR.slice(0))')
LIN_LR_POWER=$(gcloud ml-engine jobs describe $HYPERJOB --format 'value(trainingOutput.trials.hyperparameters.LIN_LR_POWER.slice(0))')
LIN_SHRINKAGE=$(gcloud ml-engine jobs describe $HYPERJOB --format 'value(trainingOutput.trials.hyperparameters.LIN_SHRINKAGE.slice(0))')
batch_size=$(gcloud ml-engine jobs describe $HYPERJOB --format 'value(trainingOutput.trials.hyperparameters.batch_size.slice(0))')

echo "Continuing to train model in $TRIALID with nnsize=$NNSIZE batch_size=$BATCHSIZE nembeds=$NEMBEDS"

CODEDIR=../loan-delinq

OUTDIR=gs://${BUCKET}/loan-delinq/training
STAGING_BUCKET="gs://${BUCKET}"
REGION="us-central1"
JOBNAME=wd_hcr_train_$(date -u +%y%m%d_%H%M)
echo $OUTDIR $REGION $JOBNAME

# directory containing trainer package in Docker image
# see Dockerfile
# gsutil -m rm -rf $OUTDIR
gcloud ml-engine jobs submit training wd_fm_$(date -u +%y%m%d_%H%M) \
  --region=$REGION \
  --module-name=trainer.task \
  --package-path=$CODEDIR/trainer \
  --job-dir=$OUTDIR \
  --staging-bucket=$STAGING_BUCKET \
  --scale-tier=STANDARD_1 \
  --runtime-version=1.8 \
  --stream-logs \
  -- \
  --output_dir=$OUTDIR \
  --input_bucket=$STAGING_BUCKET \
  --train_examples=20000 \
  --DNN_LR=$DNN_LR \
  --LIN_L1=$LIN_L1 \
  --LIN_L2=$LIN_L2 \
  --LIN_LR=$LIN_LR \
  --LIN_LR_POWER=$LIN_LR_POWER \
  --LIN_SHRINKAGE=$LIN_SHRINKAGE \
  --batch_size=$batch_size


# note --stream-logs above so that we wait for job to finish
# write output file for next step in pipeline
echo $OUTDIR > ./output.txt

# for tensorboard
#echo {\"type\":\"tensorboard\"\,\"source\":\"$OUTDIR\"} > /mlpipeline-ui-metadata.json
