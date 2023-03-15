epochs=100
net='OVANet'
ename='OpenOCT'

CUDA_VISIBLE_DEVICES='1' nohup python OVAtrain.py \
exp_name=BO_$ename num_epoch=$epochs model=$net \
source='B' target='O' train.socr_ratio=1 > octlogs/BO_$ename.log 2>&1 & \
CUDA_VISIBLE_DEVICES='2' nohup python OVAtrain.py \
exp_name=BS_$ename num_epoch=$epochs model=$net \
source='B' target='S'  > octlogs/BS_$ename.log 2>&1 & \
CUDA_VISIBLE_DEVICES='3' nohup python OVAtrain.py \
exp_name=OV_$ename num_epoch=$epochs model=$net \
source='O' target='V' train.mixup_rate=0.1 train.socr_ratio=1 > octlogs/OV_$ename.log 2>&1 & \

