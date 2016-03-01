#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. path.sh

stage=5
train_stage=0

nj=2
input_feat_dim=120   # dimension of the input features;
lstm_layer_num=4     # number of LSTM layers
lstm_cell_dim=320    # number of memory cells in every LSTM layer
max_dur=30           # max seconds of a input feat (related to GPU mem usage)
max_frames=`perl -e "print $max_dur*100"` # max frames of a input feat
dev_size=2000

num_batches_per_iter=70	# number of batches per iter
num_sequences_per_batch=10	# number of feats per batch
num_frames_per_batch=25000

add_delta=true
add_splice=false

data=data/train_fbank
labeldir=labels
dir=exp/lstm_ctc

sendMail() {
  local path=`pwd`
  local host=`hostname`
  python myScp/sendMail.py ${host}-${path}/$0
}
trap "sendMail" INT QUIT TERM EXIT


if $add_delta && $add_splice ;then
  echo "$0: add_delta=$add_delta add_splice=$add_splice, cannot set both to true" && exit 1;
fi

for f in $data/feats.scp $data/cmvn.scp $labeldir/labels.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done


# Filter train by duration
if [ $stage -le 1 ]; then
  date +"%Y-%m-%d %H:%M:%S"
  utils/copy_data_dir.sh $data ${data}_filter
  feat-to-len scp:${data}_filter/feats.scp ark,t:${data}_filter/len.scp || exit 1;
  if [ ! -z $max_dur ]; then
    echo Filtering feats by duration
    awk '{print $2}' ${data}_filter/len.scp > ${data}_filter/len.tmp || exit 1;
    paste -d " " ${data}_filter/feats.scp ${data}_filter/len.tmp > ${data}_filter/feats.scp.tmp
    paste -d " " ${data}_filter/cmvn.scp ${data}_filter/len.tmp > ${data}_filter/cmvn.scp.tmp
    cat ${data}_filter/feats.scp.tmp | \
      awk -v max_frames=$max_frames '{if($3<=max_frames) print $1 " " $2}' > ${data}_filter/feats.scp || exit 1;
    cat ${data}_filter/cmvn.scp.tmp | \
      awk -v max_frames=$max_frames '{if($3<=max_frames) print $1 " " $2}' > ${data}_filter/cmvn.scp || exit 1;
    cat ${data}_filter/feats.scp | awk '{print $1}' >${data}_filter/valid_uttlist
#   utils/filter_scp.pl ${data}_filter/valid_uttlist ${data}_filter/len.scp >${data}_filter/len.scp.tmp
#   mv ${data}_filter/len.scp.tmp ${data}_filter/len.scp
    rm -f ${data}_filter/feats.scp.tmp ${data}_filter/cmvn.scp.tmp ${data}_filter/len.tmp
  else
    echo No filtering duration, keep all feats
  fi
  # to fix labels in a naive way
  cp $labeldir/labels.scp ${data}_filter/text
  utils/fix_data_dir.sh ${data}_filter;
fi

if [ $stage -le 2 ]; then
  date +"%Y-%m-%d %H:%M:%S"
  echo Spliting feats to dev and no_dev
  # TODO: shuffle before selecting dev set
  total_size=`cat ${data}_filter/feats.scp | wc -l`
#  dev_size=$[$total_size / 10]
  utils/subset_data_dir.sh --first ${data}_filter $dev_size ${data}_filter_dev
  utils/fix_data_dir.sh ${data}_filter_dev;
  n=$[$total_size - $dev_size]
  utils/subset_data_dir.sh --last ${data}_filter $n ${data}_filter_nodev ;
  utils/fix_data_dir.sh ${data}_filter_nodev;
  echo "$0: feats total size $total_size, first $dev_size feats as dev, other $n as no_dev"
  echo 
fi

if [ $stage -le 3 ]; then
  date +"%Y-%m-%d %H:%M:%S"
  echo Generate topo
  mkdir -p $dir
  # Get the number of targets 
  if [ ! -f $dir/targets_num ]; then
    target_num=`cat $labeldir/labels.scp|
                perl -e '$max=0;while(<>){chomp;@parts=split /\s/,$_;
                foreach $part (@parts[1 .. $#parts-1]){if($part>$max){$max=$part}}};
                print $max;'`;
    target_num=$[$target_num+1]; #  #targets = #labels + 1 (the blank);
    echo $target_num >$dir/targets_num
  else
    target_num=`cat $dir/targets_num`
  fi
  echo "$0: targets number: $target_num"
  # Output the network topology
  utils/model_topo.py --input-feat-dim $input_feat_dim --lstm-layer-num $lstm_layer_num \
    --lstm-cell-dim $lstm_cell_dim --target-num $target_num --lstm-type bi \
    --fgate-bias-init 1.0 > $dir/nnet.proto || exit 1;
  echo
fi

# split and shuffle the feats
if [ $stage -le 4 ]; then
  date +"%Y-%m-%d %H:%M:%S"
  echo Splitting and shuffling
  # TODO: quesion about effiency, good or not
  # text is the label
  myScp/prep_scps.sh --add-splice "$add_splice" --add-delta "$add_delta" \
    ${data}_filter_nodev $num_sequences_per_batch $num_batches_per_iter $num_frames_per_batch $dir/data_tr
  # to split cv data into nj arks using --force-num-arks $nj
  myScp/prep_scps.sh --add-splice "$add_splice" --add-delta "$add_delta" --force-num-arks $nj \
    ${data}_filter_dev $num_sequences_per_batch 0 $num_frames_per_batch $dir/data_cv
  echo 
fi

if [ $stage -le 5 ]; then
  date +"%Y-%m-%d %H:%M:%S"
  echo Training start
  # TODO: leanring rate manipulating, including effective lr
  # TODO: progress monitoring
  # NOTE: source code modified slightly,
  # netbin/train-ctc-parallel.cc line 194 added,
  # to write #err & #ref to log, which will be parsed to show ACC.
  myScp/my_train.sh --stage 0 --report-step 350 --verbose 1 --max-epoch 10 --learn-rate 0.00004 \
    --nj $nj --num-sequence $num_sequences_per_batch --frame-num-limit $num_frames_per_batch \
    $dir/data_tr $dir/data_cv $dir || exit 1;
  echo
fi
