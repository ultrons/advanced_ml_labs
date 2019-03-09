#!/bin/bash

# Copyright 2019 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


KFP=0
while [ $# -ne 0 ]; do
    case "$1" in
       -h|--help)      echo "Usage: ./deploy.sh --modeldir <modeldir> --modelname <modelname> --modelversion <modelversion>"
                        exit
                        shift
                        ;;
       -m|--modeldir )  MODEL_LOCATION=$(gsutil ls $2/export/exporter | tail -1)
                        shift
                        ;;
       -n|--modelname )  MODEL_NAME=$2
                        shift
			;;
       -v|--modelversion )  MODEL_VERSION=$2
                        shift
			;;
       -k|--kfp)        KFP=1
                        shift
                        ;;
       *)               shift
                        ;;
    esac
done   
echo "Executing $0 $@ . ...."

if [ $KFP -eq 1 ] ; then 
     gcloud auth activate-service-account --key-file '/secret/gcp-credentials/user-gcp-sa.json'
fi
#MODEL_LOCATION=$(gsutil ls $1/export/exporter | tail -1)
#MODEL_NAME=$2
#MODEL_VERSION=$3

TFVERSION=1.8
REGION=us-central1

# create the model if it doesn't already exist
modelname=$(gcloud ml-engine models list | grep -w "$MODEL_NAME")
echo $modelname
if [ -z "$modelname" ]; then
   echo "Creating model $MODEL_NAME"
   gcloud ml-engine models create ${MODEL_NAME} --regions $REGION
else
   echo "Model $MODEL_NAME already exists"
fi

# delete the model version if it already exists
modelver=$(gcloud ml-engine versions list --model "$MODEL_NAME" | grep -w "$MODEL_VERSION")
echo $modelver
if [ "$modelver" ]; then
   echo "Deleting version $MODEL_VERSION"
   yes | gcloud ml-engine versions delete ${MODEL_VERSION} --model ${MODEL_NAME}
   sleep 10
fi


echo "Creating version $MODEL_VERSION from $MODEL_LOCATION"
gcloud ml-engine versions create ${MODEL_VERSION} \
       --model ${MODEL_NAME} --origin ${MODEL_LOCATION} \
       --runtime-version $TFVERSION

if [ $KFP -eq 1 ] ; then 
   echo $MODEL_NAME > /model.txt
   echo $MODEL_VERSION > /version.txt
fi
