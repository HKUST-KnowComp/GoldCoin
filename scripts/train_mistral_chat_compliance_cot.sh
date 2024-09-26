cd ../train
CUDA_VISIBLE_DEVICES=0 python train.py -model mistral -chat -cot -compliance -learning_rate 1e-5