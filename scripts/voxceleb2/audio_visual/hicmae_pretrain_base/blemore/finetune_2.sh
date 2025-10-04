pretrain_dataset='voxceleb2'
# dataset
finetune_dataset='blemore'
num_labels=6
# model
model_dir="hicmae_pretrain_base"
ckpt=99
# input
input_size=160
input_size_audio=256
sr=4
# parameter
lr=1e-3
epochs=100
loss='kl_div'
splits=(1 2 3 4 5 6)
val_metric='ccc'
for split in "${splits[@]}"
do
  OUTPUT_DIR="./saved/model/finetuning/${finetune_dataset}/audio_visual/${pretrain_dataset}_${model_dir}/checkpoint-${ckpt}/eval_split0${split}_lr_${lr}_epoch_${epochs}_size${input_size}_a${input_size_audio}_sr${sr}_server${server}_loss_${loss}"
  if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p $OUTPUT_DIR
  fi
# path to split files
DATA_PATH="./saved/data/${finetune_dataset}/audio_visual/split0${split}"
# path to pre-trained model
MODEL_PATH="./saved/model/pretraining/${pretrain_dataset}/audio_visual/${model_dir}/checkpoint-${ckpt}.pth"

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port 12497 \
    run_class_finetuning_regression_av.py \
    --model avit_dim512_patch16_160_a256 \
    --data_set ${finetune_dataset} \
    --nb_classes ${num_labels} \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 32 \
    --num_sample 1 \
    --input_size ${input_size} \
    --input_size_audio ${input_size_audio} \
    --short_side_size ${input_size} \
    --depth 10 \
    --depth_audio 10 \
    --fusion_depth 2 \
    --save_ckpt_freq 1000 \
    --num_frames 16 \
    --sampling_rate ${sr} \
    --opt adamw \
    --lr ${lr} \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs ${epochs} \
    --dist_eval \
    --test_num_segment 2 \
    --test_num_crop 2 \
    --loss ${loss} \
    --num_workers 16 \
    --val_metric ${val_metric} \
    >${OUTPUT_DIR}/nohup.out 2>&1
done
echo "Done!"



