#!/bin/bash

input=$1
name=$2
lang=$3
rm=$4

#expected structure: .${data_dir}/${input}/[train or valid]/[train or valid].[0 or 1]

prepr_dir=/Users/valentin/BThesis/Preprocessing
current_dir=$(pwd)
data_dir=/Users/valentin/BThesis/data

cd $prepr_dir

mkdir -p ${data_dir}/tmp/${name}/tok/train
mkdir -p ${data_dir}/tmp/${name}/tok/valid
mkdir -p ${data_dir}/tmp/${name}/sc/train
mkdir -p ${data_dir}/tmp/${name}/sc/valid
mkdir -p model/${name}
mkdir -p ${data_dir}/${name}/train
mkdir -p ${data_dir}/${name}/valid

##TOKENIZE

echo "" > ${data_dir}/tmp/${name}/corpus.tok.0
for f in ${data_dir}/${input}/train/*\.0
do
cat $f | perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l ${lang} > ${data_dir}/${name}/train/${f##*/}
done
for f in ${data_dir}/${input}/valid/*\.0
do
cat $f | perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l ${lang} > ${data_dir}/${name}/valid/${f##*/}
done



echo "" > ${data_dir}/tmp/${name}/corpus.tok.1
for f in ${data_dir}/${input}/train/*\.1
do
cat $f | perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l ${lang} > ${data_dir}/${name}/train/${f##*/}
done
for f in ${data_dir}/${input}/valid/*\.1
do
cat $f | perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l ${lang} > ${data_dir}/${name}/valid/${f##*/}
done



cd $current_dir
python3 preprocess_style.py \
-train_style1 ${data_dir}/${name}/train/train.0 \
-train_style2 ${data_dir}/${name}/train/train.1 \
-train_style1_rm ${data_dir}/${name}/train/train-del-${rm}.0 \
-train_style2_rm ${data_dir}/${name}/train/train-del-${rm}.1 \
-valid_style1 ${data_dir}/${name}/valid/valid.0 \
-valid_style2 ${data_dir}/${name}/valid/valid.1 \
-valid_style1_rm ${data_dir}/${name}/valid/valid-del-${rm}.0 \
-valid_style2_rm ${data_dir}/${name}/valid/valid-del-${rm}.1 \
-save_data ${data_dir}/${name}/model_prepr

rm -r ${data_dir}/tmp/${name}/