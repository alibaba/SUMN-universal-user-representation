# SUMN-universal-user-representation

Exploiting Behavioral Consistence for Universal User Representation (AAAI 2021)

By Jie Gu*, Feng Wang*, Qinghui Sun, Zhiquan Ye, Xiaoxiao Xu, Jingmin Chen, Jun Zhang

arxiv: https://arxiv.org/abs/2012.06146

## Introduction
In this paper, we focus on universal (general-purpose) user representation. The obtained universal representations are expected to contain rich information, and be applicable to various downstream applications without further modifications (e.g., user preference prediction and user profiling). Accordingly, we can be free from the heavy work of training task-specific models for every downstream task as in previous works.

## Usage
The project is developed based on [Alibaba Cloud](https://www.alibabacloud.com/help/en/): [PAI](https://www.alibabacloud.com/product/machine-learning) tensorflow, [MaxCompute](https://www.alibabacloud.com/product/maxcompute) (a data processing platform for large-scale data warehousing) and [OSS](https://www.alibabacloud.com/product/object-storage-service) (a storage service). 
The related APIs and functions are: tf.python_io.TableWriter in dumper.py, tf.data.TableRecordDataset in loader.py, tf.gfile in train.py, gfile in inference.py, gfile in helper.py, and set_dist_env in env.py. You can revise these APIs and functions if needed. 

Script for training the representation model
```
rm model.tar.gz
tar -czf model.tar.gz ./data_dumper ./data_loader ./main ./model ./util

ITERATION=1000000
SNAPSHOT=100000
TARGET_LENGTH=500
LEARNING_RATE=0.001
DROPOUT=0.1
BATCH_SIZE=128
MODEL_DIM=256
WORD_DIMENSION=256
POOL=mhop
CHECKPOINT_PATH=your_own_oss_directory/drop${DROPOUT}_lr${LEARNING_RATE}_tarlen${TARGET_LENGTH}_pool${POOL}_dim256_mlp512_b${BATCH_SIZE}

odpscmd -e "use your_own_maxcompute_project; pai \
        -name tensorflow180 -project algo_public \
        -Dscript=\"file://`pwd`/model.tar.gz\" \
        -Dtables=\"odps://your_own_maxcompute_project/tables/your_own_maxcompute_table_for_model_training\" \
        -DentryFile=\"main/train.py\" \
        -DgpuRequired=\"100\"\
        -Dbuckets=\"your_own_oss_buckets\" \
        -DuserDefinedParameters='--max_steps=${ITERATION} --snapshot=${SNAPSHOT} --checkpoint_dir=${CHECKPOINT_PATH} \
            --target_length=${TARGET_LENGTH} --learning_rate=${LEARNING_RATE} --dropout=${DROPOUT} --batch_size=${BATCH_SIZE} \
            --model_dim=${MODEL_DIM} --word_emb_dim=${WORD_DIMENSION} --pooling=${POOL} \
            --max_query_per_week=900  --max_words_per_query=24' \
        "
```

Script for representation inference
```
rm model.tar.gz
tar -czf model.tar.gz ./data_dumper ./data_loader ./main ./model ./util

INPUT_TABLE=your_own_maxcompute_table_for_inferring_inputs
OUTPUT_TABLE=your_own_maxcompute_table_for_inferring_outputs
CHECKPOINT_PATH=your_own_oss_directory/drop0.1_lr0.001_tarlen1000_poolmax_dim256_mlp512_b128/
STEP=1000000

odpscmd -e "use your_own_maxcompute_project; pai \
        -name tensorflow180 -project algo_public \
        -Dscript=\"file://`pwd`/model.tar.gz\" \
        -Dtables=\"odps://your_own_maxcompute_project/tables/${INPUT_TABLE}\" \
        -Doutputs=\"odps://your_own_maxcompute_project/tables/${OUTPUT_TABLE}/version=drop0.1_lr0.001_tarlen1000_poolmax_dim256_mlp512_b128_100w\" \
        -DentryFile=\"main/inference.py\" \
        -Dbuckets=\"your_own_oss_buckets\" \
        -DuserDefinedParameters=\"--checkpoint_dir='$CHECKPOINT_PATH' --step=$STEP \" \
        -Dcluster='{\"worker\":{\"count\":64,\"cpu\":200,\"memory\":4096,\"gpu\":50}}' \
        "
```

## License
See LICENSE for details.

## Citation
If you find this repo useful in your research, please consider citing the paper:
```
@inproceedings{SUMN_user_representation,
  author    = {Jie Gu and Feng Wang and Qinghui Sun and Zhiquan Ye and Xiaoxiao Xu and Jingmin Chen and Jun Zhang},
  title     = {Exploiting Behavioral Consistence for Universal User Representation},
  booktitle = {Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI}
               2021},
  pages     = {4063--4071},
  year      = {2021}
}
```
