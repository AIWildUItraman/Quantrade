export CUDA_VISIBLE_DEVICES=6,7,8,9
model_name=TimesNet
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /home/mengxiaosen/mxs/workspace/Quantrade/data/NEIRO/ \
  --data_path NERIRO2hLabel.csv \
  --model_id klinemodel \
  --model $model_name \
  --data kline \
  --features M \
  --seq_len 20 \
  --enc_in 6 \
  --batch_size 256 \
  --d_model 64 \
  --d_ff 128 \
  --e_layers 3 \
  --top_k 3 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10 \
  --dropout 0.1 \
  --num_workers 4 \
  --use_multi_gpu \
  --devices 0,1,2,3