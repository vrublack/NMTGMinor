#!/bin/bash


input=$1
name=$2
lang=$3

#expected structure: /data/${input}/[train or valid]/[train or valid].[0 or 1]

basedir=/Users/valentin/BThesis/Preprocessing
cd $basedir

mkdir -p tmp/${name}/tok/train
mkdir -p tmp/${name}/tok/valid
mkdir -p tmp/${name}/sc/train
mkdir -p tmp/${name}/sc/valid
mkdir -p model/${name}
mkdir -p ../data/${name}/train
mkdir -p ../data/${name}/valid

##TOKENIZE

echo "" > tmp/${name}/corpus.tok.0
for f in ../data/${input}/train/*\.0
do
cat $f | perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l ${lang} > tmp/${name}/tok/train/${f##*/}
cat tmp/${name}/tok/train/${f##*/} >> tmp/${name}/corpus.tok.0
done
for f in ../data/${input}/valid/*\.0
do
cat $f | perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l ${lang} > tmp/${name}/tok/valid/${f##*/}
done



echo "" > tmp/${name}/corpus.tok.1
for f in ../data/${input}/train/*\.1
do
cat $f | perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l ${lang} > tmp/${name}/tok/train/${f##*/}
cat tmp/${name}/tok/train/${f##*/} >> tmp/${name}/corpus.tok.1
done
for f in ../data/${input}/valid/*\.1
do
cat $f | perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l ${lang} > tmp/${name}/tok/valid/${f##*/}
done



##SMARTCASE


mosesdecoder/scripts/recaser/train-truecaser.perl --model model/${name}/truecase-model.0 --corpus tmp/${name}/corpus.tok.0
mosesdecoder/scripts/recaser/train-truecaser.perl --model model/${name}/truecase-model.1 --corpus tmp/${name}/corpus.tok.1

for set in valid train
do
for f in tmp/${name}/tok/$set/*\.0
do
cat $f | mosesdecoder/scripts/recaser/truecase.perl --model model/${name}/truecase-model.0 > tmp/${name}/sc/$set/${f##*/}
done
done

for set in valid train
do
for f in tmp/${name}/tok/$set/*\.1
do
cat $f | mosesdecoder/scripts/recaser/truecase.perl --model model/${name}/truecase-model.1 > tmp/${name}/sc/$set/${f##*/}
done
done

echo "" > tmp/${name}/corpus.sc.0
for f in tmp/${name}/sc/train/*\.0
do
cat $f >> tmp/${name}/corpus.sc.0
done

echo "" > tmp/${name}/corpus.sc.1
for f in tmp/${name}/sc/train/*\.1
do
cat $f >> tmp/${name}/corpus.sc.1
done

##BPE


subword-nmt/subword_nmt/learn_joint_bpe_and_vocab.py --input tmp/${name}/corpus.sc.0 tmp/${name}/corpus.sc.1 -s 40000 -o model/${name}/codec --write-vocabulary model/${name}/voc.0 model/${name}/voc.1


for set in valid train
do
for f in tmp/${name}/sc/$set/*\.0
do
echo $f
subword-nmt/subword_nmt/apply_bpe.py -c model/${name}/codec --vocabulary model/${name}/voc.0 --vocabulary-threshold 50 < $f > ../data/${name}/$set/${f##*/}
done
done

for set in valid train
do
for f in tmp/${name}/sc/$set/*\.1
do
echo $f
subword-nmt/subword_nmt/apply_bpe.py -c model/${name}/codec --vocabulary model/${name}/voc.1 --vocabulary-threshold 50 < $f > ../data/${name}/$set/${f##*/}
done
done


python preprocess_style.py -train_src ../data/${name}/train/train.0 -train_tgt ../data/${name}/train/train.1 -valid_src ../data/${name}/valid/valid.0 -valid_tgt ../data/${name}/valid/valid.0 -save_data data/${name}

rm -r tmp/${name}/