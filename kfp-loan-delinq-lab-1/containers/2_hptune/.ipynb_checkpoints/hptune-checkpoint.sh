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
       -h|--help)      echo "Usage: ./train.sh --bucket <bucket-name>"
                        exit
                        shift
                        ;;
       -b|--bucket)     BUCKET=$2
                        shift
                        ;;
       *)               shift
                        ;;
    esac
done   
                     



TFVERSION=1.8
REGION=us-central1
#echo "$BUCKET, $RUN_MODE"
#exit

# directory containing trainer package in Docker image
# see Dockerfile
CODEDIR=/loan-delinq

OUTDIR=gs://${BUCKET}/loan-delinq/hyperparam
STAGING_BUCKET="gs://${BUCKET}"
REGION="us-central1"
JOBNAME=wd_hcr_hptuning_$(date -u +%y%m%d_%H%M)
echo $OUTDIR $REGION $JOBNAME

# {gsutil -m rm -rf $OUTDIR } || { echo "No object was deleted" }
 # --python-version=3.5 \

gcloud ml-engine jobs submit training $JOBNAME \
  --region=$REGION \
  --module-name=trainer.task \
  --package-path=$CODEDIR/trainer \
  --job-dir=$OUTDIR \
  --staging-bucket=$STAGING_BUCKET \
  --config=$CODEDIR/hpconfig.yaml \
  --runtime-version=1.8 \
  --stream-logs \
  -- \
  --output_dir=$OUTDIR \
  --input_bucket=$STAGING_BUCKET \
  --eval_steps=10 \
  --train_examples=20000


# note --stream-logs above so that we wait for job to finish
# write output file for next step in pipeline
echo $JOBNAME > /output.txt
