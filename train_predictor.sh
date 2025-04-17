PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port=1234 /fujiayi/IPC-Dehaze/basicsr/train.py -opt options/02train_predictor.yml --launcher pytorch 




