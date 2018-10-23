# to run on cluster

mkdir -p /tmp/vrublack
cp -n /data/ASR5/vrublack/emb/glove.twitter.27B.200d.txt /tmp/vrublack/glove.twitter.27B.200d.txt
cp -r -n /data/ASR5/vrublack/data/yelp /tmp/vrublack/yelp

DATA_PATH=/tmp/vrublack/yelp/valid
# without BPE!
REF_PATH=/tmp/vrublack/yelp/valid
SST_PATH=/home/vrublack/SST-RNN
EMB_PATH=/tmp/vrublack/glove.twitter.27B.200d.txt


models=( "$@" )
arraylength=${#models[@]}

OUT_FNAMES=()
REF_FNAMES=()

for (( i=0; i<${arraylength}; i++ ));
do
    for (( t=1; t<=2; t++ ));
    do
        if [ $t -eq 1 ]; then
           src_t=1
        else
           src_t=0
        fi
        printf -v FNAME "out/mout_%s_target_%d" "${models[$i]}" $t
        python3 ../translate.py -model ../save/${models[$i]} -src ${DATA_PATH}/valid-mid.$src_t -target_style $t -out $FNAME -max_sent_length 100 -remove_bpe -gpu 0
        OUT_FNAMES+=($FNAME)
        REF_FNAMES+=(${REF_PATH}/valid-mid.$src_t)
    done
done


python3 ${SST_PATH}/predict_style_transfer.py \
--emb_path $EMB_PATH \
--style_model ${SST_PATH}/save/yelp_sst_rnn.pt \
--language_model ${SST_PATH}/base-1/save/yelp-lm.pt \
--idx ${SST_PATH}/save/index-glove.twitter.27B.200d.txt \
--emsize 200 \
--output_path ./out \
--cuda \
--inputs_transferred "${OUT_FNAMES[@]}" \
--inputs_original "${REF_FNAMES[@]}" \