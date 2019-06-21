## 10分钟入门

### 准备数据
准备src和tgt数据，每行一句话，词之间用空格隔开，nfsdata上的样例数据路径如下
/data/nfsdata/data/yuxian/datasets/gec/csc/toy_train.src
/data/nfsdata/data/yuxian/datasets/gec/csc/toy_train.tgt
/data/nfsdata/data/yuxian/datasets/gec/csc/toy_valid.src
/data/nfsdata/data/yuxian/datasets/gec/csc/toy_valid.tgt
/data/nfsdata/data/yuxian/datasets/gec/csc/toy_test.src
/data/nfsdata/data/yuxian/datasets/gec/csc/toy_test.tgt

准备srcdict
srcdict=/data/nfsdata/data/yuxian/datasets/gec/toy_dict.txt

###数据预处理
调用preprocess.py，将train和valid转化为bin和idx文件，将test转化为raw data。
```
trainpref=/data/nfsdata/data/yuxian/datasets/gec/csc/toy_train
validpref=/data/nfsdata/data/yuxian/datasets/gec/csc/toy_valid
testpref=/data/nfsdata/data/yuxian/datasets/gec/csc/toy_test

# 存放fairseq训练数据的路径
OUT=$trainpref

DATA_BIN=$OUT/data_bin
DATA_RAW=$OUT/data_raw
mkdir -p $DATA_BIN
mkdir -p $DATA_RAW

# 如果要重新生成数据，需要删除原来的dict文件
rm ${DATA_BIN}/dict.src.txt
rm ${DATA_BIN}/dict.tgt.txt
rm ${DATA_RAW}/dict.src.txt
rm ${DATA_RAW}/dict.tgt.txt

# 生成二进制的train/valid
python preprocess.py \
--trainpref $trainpref \
--validpref $validpref \
--source-lang src --target-lang tgt \
--padding-factor 1 \
--joined-dictionary \
--copy-ext-dict \
--destdir ${DATA_BIN} \
--workers 1 \
--srcdict $srcdict \

# 生成raw(txt)的test
srcdict=${DATA_BIN}/dict.src.txt
python preprocess.py \
--source-lang src --target-lang tgt  \
--padding-factor 1 \
--joined-dictionary \
--copy-ext-dict \
--testpref ${testpref} \
--destdir ${DATA_RAW} \
--dataset-impl raw \
--srcdict ${srcdict}
```

### 训练
```
device=0

DATAPATHS=/data/nfsdata/data/yuxian/datasets/gec/csc/toy_train/data_bin
MODELS=/data/nfsdata/data/yuxian/train_logs/toy_train

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
--num-workers 4 \
--dataset-impl lazy \
--copy-attention --copy-attention-heads 1 --copy-attention-dropout 0.0 --copy-ext-dict \

```
注意需要调整参数如dropout，embed-dim，heads

### 生成
```
python generate.py $DATA_RAW \
  --gen-subset test \
  --path ${MODELS}/checkpoint_best.pt \
  --dataset-impl raw \
  --beam 5 \
  --remove-bpe \
  --batch-size 128 \
  --copy-ext-dict \
```

