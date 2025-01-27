export PYTHONPATH=/home/alee00/Align3R

# Training set
CUDA_VISIBLE_DEVICES=1 python infer_training_set.py --a=0 --b=-1 --dataset_name=Tartanair &\
# CUDA_VISIBLE_DEVICES=1 python infer_training_set.py --a=0 --b=-1 --dataset_name=spring &\
# CUDA_VISIBLE_DEVICES=1 python infer_training_set.py --a=0 --b=-1 --dataset_name=SceneFlow &\
CUDA_VISIBLE_DEVICES=1 python infer_training_set.py --a=0 --b=-1 --dataset_name=Vkitti &\
# CUDA_VISIBLE_DEVICES=1 python infer_training_set.py --a=0 --b=-1 --dataset_name=PointOdyssey

# Test set
# CUDA_VISIBLE_DEVICES=1 python infer_test_set.py --a=0 --b=10000 --dataset_name=bonn &\
# # CUDA_VISIBLE_DEVICES=1 python infer_test_set.py --a=0 --b=10000 --dataset_name=davis &\ # Disabled due to missing data
# # CUDA_VISIBLE_DEVICES=1 python infer_test_set.py --a=0 --b=10000 --dataset_name=sintel &\ # Disabled due to missing data
# CUDA_VISIBLE_DEVICES=1 python infer_test_set.py --a=0 --b=10000 --dataset_name=tum
