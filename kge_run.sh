#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

CODE_PATH=codes
DATA_PATH=data
SAVE_PATH=models

#The first four parameters must be provided
MODE=$1
MODEL=$2
DATASET=$3
GPU_DEVICE=$4
SAVE_ID=$5

FULL_DATA_PATH=$DATA_PATH/$DATASET

#Only used in training
BATCH_SIZE=$6
NEGATIVE_SAMPLE_SIZE=$7
HIDDEN_DIM=$8
GAMMA=$9
ALPHA=${10}
LEARNING_RATE=${11}
MAX_STEPS=${12}
TEST_BATCH_SIZE=${13}
AUTOENCODER_STATUS=$(echo ${14} | tr '[:upper:]' '[:lower:]')
AUTOENCODER_LAMBDA=${15}
AUTOENCODER_HIDDEN_DIM=${16}

if [ "$AUTOENCODER_STATUS" = "true" ] || [ "$AUTOENCODER_STATUS" = "1" ]; then
    AUTOENCODER_FLAG="--autoencoder_flag"
else
    AUTOENCODER_FLAG=""
fi

# SAVE is modified, and different from original code
# SAVE_ID is now useless
SAVE=$SAVE_PATH/"$MODEL"_"$DATASET"_dim"$HIDDEN_DIM"
# if Autoencoder is used, add it to the SAVE name
if [ "$AUTOENCODER_STATUS" = "true" ] || [ "$AUTOENCODER_STATUS" = "1" ]; then
    SAVE=$SAVE"_autoencoder"
fi

if [ $MODE == "train" ]
then

echo "Start Training......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u kge_train.py --do_train \
    --cuda \
    --do_valid \
    --do_test \
    --data_path $FULL_DATA_PATH \
    --model $MODEL \
    -n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $HIDDEN_DIM \
    -g $GAMMA -a $ALPHA -adv \
    -lr $LEARNING_RATE --max_steps $MAX_STEPS \
    -save $SAVE --test_batch_size $TEST_BATCH_SIZE \
    $AUTOENCODER_FLAG \
    --autoencoder_lambda $AUTOENCODER_LAMBDA \
    --autoencoder_hidden_dim $AUTOENCODER_HIDDEN_DIM \
    ${17} ${18} ${19} ${20} ${21} ${22} ${23} ${24} ${25}

elif [ $MODE == "valid" ]
then

echo "Start Evaluation on Valid Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_valid --cuda -init $SAVE --save_path fb15k_237
   
elif [ $MODE == "test" ]
then

echo "Start Evaluation on Test Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_test --cuda -init $SAVE  --save_path fb15k_237

else
   echo "Unknown MODE" $MODE
fi