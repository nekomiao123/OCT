epochs=100
net='OVANet'
ename='OpenOCT2'

CUDA_VISIBLE_DEVICES='6' nohup python train.py \
exp_name=BO_$ename num_epoch=$epochs model=$net \
source='B' target='O' > logs/BO_$ename.log 2>&1 & \
CUDA_VISIBLE_DEVICES='4' nohup python train.py \
exp_name=BS_$ename num_epoch=$epochs model=$net \
source='B' target='S'  > logs/BS_$ename.log 2>&1 & \
CUDA_VISIBLE_DEVICES='5' nohup python train.py \
exp_name=OV_$ename num_epoch=$epochs model=$net \
source='O' target='V' train.mixup_rate=0.1 > logs/OV_$ename.log 2>&1 & \

