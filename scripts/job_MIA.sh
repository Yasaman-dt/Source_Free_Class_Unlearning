DATASET=$1 #"cifar10", "cifar100", "TinyImageNet"
METHOD=$2
N_MODEL=$3
SOURCE=$4
LR=$5
GPU=$6
MODEL=$7

SCREEN_NAME="${DATASET}_${METHOD}_${N_MODEL}_${SOURCE}"

screen -S $SCREEN_NAME -dm bash -c "
source ~/.bashrc
conda activate /projets/Zdehghani/torch_env
cd /projets/Zdehghani/Source_Free_Class_Unlearning/MIA_code
CUDA_VISIBLE_DEVICES=$GPU \
python checking_MIA.py \
    --method $METHOD \
    --model $MODEL \
    --n_model $N_MODEL \
    --source $SOURCE \
    --lr $LR \
    --dataset $DATASET
"