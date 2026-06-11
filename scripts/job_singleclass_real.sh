DATASET=$1 #"cifar10", "cifar100", "TinyImageNet"
METHOD=$2
lr=$3
N_MODEL=$4
SAMPLE_PER_CLASS=$5
gpu=$6
epoch=$7
MODEL=$8

SCREEN_NAME="real_${DATASET}_${METHOD}_${N_MODEL}_${lr}_${epoch}_${SAMPLE_PER_CLASS}_${MODEL}.sh"

screen -S $SCREEN_NAME -dm bash -c "
source ~/.bashrc
conda activate /projets/Zdehghani/torch_env
cd /projets/Zdehghani/Source_Free_Class_Unlearning
CUDA_VISIBLE_DEVICES=$gpu \
python -m main_files.main_real  \
    --dataset $DATASET \
    --mode CR \
    --cuda 0 \
    --save_model \
    --save_df \
    --run_unlearn  \
    --num_workers 4 \
    --method $METHOD \
    --model $MODEL \
    --bsize 1024 \
    --lr $lr \
    --epochs $epoch  \
    --patience 50  \
    --samples_per_class $SAMPLE_PER_CLASS  \
    --forget_mode single \
    --n_model $N_MODEL
"
