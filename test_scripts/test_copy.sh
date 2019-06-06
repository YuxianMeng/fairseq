#!/usr/bin/env bash

#####

srcdict=/data/nfsdata/data/yuxian/datasets/gec/toy_dict.txt
ofile_prefix=/data/nfsdata2/yuxian/datasets/gec/toy
validpref=/data/nfsdata/data/yuxian/datasets/gec/csc/test
testpref=/data/nfsdata/data/yuxian/datasets/gec/csc/test
DATAPATHS=""

##### preprocess train/valid
for idx in 1; do
    trainpref=$ofile_prefix$idx

    OUT=$trainpref

    DATA_BIN=$OUT/data_bin
    DATA_RAW=$OUT/data_raw
    mkdir -p $DATA_BIN
    mkdir -p $DATA_RAW


    rm  $DATA_BIN/dict.tgt.txt
    python preprocess.py \
    --srcdict $srcdict \
    --validpref $validpref \
    --source-lang src --target-lang tgt \
    --padding-factor 1 \
    --joined-dictionary \
    --copy-ext-dict \
    --trainpref $trainpref \
    --destdir $DATA_BIN \
    --workers 1 \

    chmod -R 777 $OUT
    DATAPATHS=${DATAPATHS}" "${DATA_BIN}
done


##### Train
device=0

DATAPATHS="/data/nfsdata2/yuxian/datasets/gec/toy1/data_bin"
MODELS=/data/nfsdata/data/yuxian/train_logs/zh_toy2

CUDA_VISIBLE_DEVICES=$device python train.py $DATAPATHS \
--source-lang src --target-lang tgt \
--save-dir ${MODELS} \
--max-epoch 1000 \
--batch-size 4 \
--max-tokens 1000 \
--train-subset train \
--valid-subset valid \
--arch transformer \
--encoder-layers 3 --decoder-layers 3 \
--lr-scheduler triangular --lr 0.002 --max-lr 0.004 --lr-period-updates 100 \
--clip-norm 2  --lr-shrink 0.99 --shrink-min \
--dropout 0.2 --relu-dropout 0.0 --attention-dropout 0.0 \
--encoder-embed-dim 128 --decoder-embed-dim 128 \
--max-target-positions 512 --max-source-positions 512 \
--encoder-ffn-embed-dim 512 --decoder-ffn-embed-dim 512 \
--encoder-attention-heads 1 --decoder-attention-heads 1 \
--share-all-embeddings \
--num-workers 1 \
--dataset-impl lazy \
--save-interval 500 --keep-interval-updates 1  --disable-validation \
--copy-attention --copy-attention-heads 1 --copy-attention-dropout 0.0 --copy-ext-dict \


# Generate
split=false
testpref=/data/nfsdata/data/yuxian/datasets/gec/csc/test_toy
DATA_RAW=/data/nfsdata/data/yuxian/train_logs/data_raw
golden=/data/nfsdata2/yuxian/datasets/source_gold_test_yuxian.txt

RESULTS="/data/nfsdata/data/yuxian/train_logs/toy"
mkdir -p $RESULTS
rm -f ${DATA_RAW}/dict.src.txt
rm -f ${DATA_RAW}/dict.tgt.txt
python preprocess.py \
--source-lang src --target-lang tgt  \
--padding-factor 1 \
--joined-dictionary \
--testpref ${testpref} \
--destdir ${DATA_RAW} \
--dataset-impl raw \
--srcdict ${srcdict}

python generate.py $DATA_RAW \
  --gen-subset test \
  --path $MODELS/checkpoint_last.pt \
  --dataset-impl raw \
  --beam 5 \
  --remove-bpe \
  --batch-size 128 \
  --copy-ext-dict \
  >$RESULTS/epoch_last_predict.split

## 将fairseq导出的.split文件按idx sort
cat $RESULTS/epoch_last_predict.split | grep "^H" | python ./gec_scripts/sort.py 12 \
$RESULTS/epoch_last_predict.split2


