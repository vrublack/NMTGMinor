#!/bin/bash

input=$1
name=$2
lang=$3

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
python3 preprocess_style.py -join_vocab -train_style1 ${data_dir}/${name}/train/train.0 -train_style2 ${data_dir}/${name}/train/train.1 -valid_style1 ${data_dir}/${name}/valid/valid.0 -valid_style2 ${data_dir}/${name}/valid/valid.0 -save_data ${data_dir}/${name}/model_prepr

rm -r ${data_dir}/tmp/${name}/