# to run on local computer

DATA_PATH=/Users/valentin/BThesis/data/yelp/valid
REF_PATH=/Users/valentin/BThesis/data/yelp/valid
SST_PATH=/Users/valentin/PycharmProjects/SST-RNN-Pytorch
EMB_PATH=/Users/valentin/Downloads/glove.twitter.27B/glove.twitter.27B.200d.txt

models_01=$1
models_10=$2



printf -v FNAME_01 "out/mout_%s_target_2" ${models_01}
python3 ../translate.py -model ../save/${models_01} -src ${DATA_PATH}/valid-mid.0 -out ${FNAME_01} -max_sent_length 30

printf -v FNAME_10 "out/mout_%s_target_1" ${models_10}
python3 ../translate.py -model ../save/${models_10} -src ${DATA_PATH}/valid-mid.1 -out ${FNAME_10} -max_sent_length 30


python3 ${SST_PATH}/predict_style_transfer.py \
--emb_path $EMB_PATH \
--style_model ${SST_PATH}/save/yelp_sst_rnn.pt \
--language_model ${SST_PATH}/base-1/save/yelp-lm.pt \
--idx ${SST_PATH}/save/index-glove.twitter.27B.200d.txt \
--emsize 200 \
--output_path ./out \
--inputs_transferred ${FNAME_01} ${FNAME_10} \
--inputs_original ${DATA_PATH}/valid-mid.0 ${DATA_PATH}/valid-mid.1