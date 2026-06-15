import os

cfg = 'configs/config1.yaml'

process = 'ttbar'
ckpts = 'ckpts_{}'.format(process)

os.system('mkdir -p {}'.format(ckpts))

cmd = 'python train.py --data data_{} --ckpts {} --cfg {}'.format(process, ckpts, cfg)

print(cmd)
os.system(cmd)
