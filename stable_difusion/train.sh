LAUNCH_TRAINING(){

accelerate config default
pretrained_model_name_or_path='prs-eth/marigold-v1-0'
root_path='dataset_160x120'
output_dir='outputs'
train_batch_size=1
num_train_epochs=100
gradient_accumulation_steps=8
learning_rate=1e-5
lr_warmup_steps=0
dataloader_num_workers=4
tracker_project_name='Custom_StableDiffusion'


CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="fp16" train_sd.py \
                  --pretrained_model_name_or_path $pretrained_model_name_or_path \
                  --dataset_path $root_path\
                  --output_dir $output_dir \
                  --train_batch_size $train_batch_size \
                  --num_train_epochs $num_train_epochs \
                  --gradient_accumulation_steps $gradient_accumulation_steps\
                  --gradient_checkpointing \
                  --learning_rate $learning_rate \
                  --lr_warmup_steps $lr_warmup_steps \
                  --dataloader_num_workers $dataloader_num_workers \
                  --tracker_project_name $tracker_project_name \
                  --enable_xformers_memory_efficient_attention 

}



LAUNCH_TRAINING
