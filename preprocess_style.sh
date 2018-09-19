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
cat $f | perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l ${lang} > ${data_dir}/tmp/${name}/tok/train/${f##*/}
cat ${data_dir}/tmp/${name}/tok/train/${f##*/} >> ${data_dir}/tmp/${name}/corpus.tok.0
done
for f in ${data_dir}/${input}/valid/*\.0
do
cat $f | perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l ${lang} > ${data_dir}/tmp/${name}/tok/valid/${f##*/}
done



echo "" > ${data_dir}/tmp/${name}/corpus.tok.1
for f in ${data_dir}/${input}/train/*\.1
do
cat $f | perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l ${lang} > ${data_dir}/tmp/${name}/tok/train/${f##*/}
cat ${data_dir}/tmp/${name}/tok/train/${f##*/} >> ${data_dir}/tmp/${name}/corpus.tok.1
done
for f in ${data_dir}/${input}/valid/*\.1
do
cat $f | perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l ${lang} > ${data_dir}/tmp/${name}/tok/valid/${f##*/}
done



##SMARTCASE


mosesdecoder/scripts/recaser/train-truecaser.perl --model model/${name}/truecase-model.0 --corpus ${data_dir}/tmp/${name}/corpus.tok.0
mosesdecoder/scripts/recaser/train-truecaser.perl --model model/${name}/truecase-model.1 --corpus ${data_dir}/tmp/${name}/corpus.tok.1

for set in valid train
do
for f in ${data_dir}/tmp/${name}/tok/$set/*\.0
do
cat $f | mosesdecoder/scripts/recaser/truecase.perl --model model/${name}/truecase-model.0 > ${data_dir}/tmp/${name}/sc/$set/${f##*/}
done
done

for set in valid train
do
for f in ${data_dir}/tmp/${name}/tok/$set/*\.1
do
cat $f | mosesdecoder/scripts/recaser/truecase.perl --model model/${name}/truecase-model.1 > ${data_dir}/tmp/${name}/sc/$set/${f##*/}
done
done

echo "" > ${data_dir}/tmp/${name}/corpus.sc.0
for f in ${data_dir}/tmp/${name}/sc/train/*\.0
do
cat $f >> ${data_dir}/tmp/${name}/corpus.sc.0
done

echo "" > ${data_dir}/tmp/${name}/corpus.sc.1
for f in ${data_dir}/tmp/${name}/sc/train/*\.1
do
cat $f >> ${data_dir}/tmp/${name}/corpus.sc.1
done

##BPE


subword-nmt/subword_nmt/learn_joint_bpe_and_vocab.py --input ${data_dir}/tmp/${name}/corpus.sc.0 ${data_dir}/tmp/${name}/corpus.sc.1 -s 40000 -o model/${name}/codec --write-vocabulary model/${name}/voc.0 model/${name}/voc.1


for set in valid train
do
for f in ${data_dir}/tmp/${name}/sc/$set/*\.0
do
echo $f
subword-nmt/subword_nmt/apply_bpe.py -c model/${name}/codec --vocabulary model/${name}/voc.0 --vocabulary-threshold 50 < $f > ${data_dir}/${name}/$set/${f##*/}
done
done

for set in valid train
do
for f in ${data_dir}/tmp/${name}/sc/$set/*\.1
do
echo $f
subword-nmt/subword_nmt/apply_bpe.py -c model/${name}/codec --vocabulary model/${name}/voc.1 --vocabulary-threshold 50 < $f > ${data_dir}/${name}/$set/${f##*/}
done
done


cd $current_dir
python3 preprocess_style.py -train_src ${data_dir}/${name}/train/train.0 -train_tgt ${data_dir}/${name}/train/train.1 -valid_src ${data_dir}/${name}/valid/valid.0 -valid_tgt ${data_dir}/${name}/valid/valid.0 -save_data ${data_dir}/${name}/model_prepr

rm -r ${data_dir}/tmp/${name}/