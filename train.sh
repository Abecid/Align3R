#!/bin/bash
export PYTHONPATH=$PYTHONPATH:Align3R

dataset_root="/scratch/partial_datasets/align3r/data"
CUDA_VISIBLE_DEVICES="2,3" torchrun --master_port=2569 --nproc_per_node 2 tool/train.py \
                          --train_dataset=" + 1_2500 @ PointodysseyDatasets(split='train', ROOT='$dataset_root/PointOdyssey_proc/train', aug_crop=16, aug_f=True, resolution=[(512, 288),  (512, 336), (512, 256)], transform=ColorJitter, depth_prior_name='moge') + 1_000 @ SpringDatasets(split='train', ROOT='$dataset_root/spring_proc/train', aug_crop=16, aug_f=True, resolution=[(512, 288),  (512, 336), (512, 256)], transform=ColorJitter, depth_prior_name='moge')+ 5_000 @ SceneFlowDatasets(split='train', ROOT='$dataset_root/SceneFlow/', aug_crop=16, aug_f=True, resolution=[(512, 288), (512, 336), (512, 256)], transform=ColorJitter, depth_prior_name='moge') + 3_000 @ VkittiDatasets(split='train', ROOT='$dataset_root/vkitti_2.0.3_proc/', aug_crop=16, aug_f=True, resolution=[(512, 288), (512, 336), (512, 256)], transform=ColorJitter, depth_prior_name='moge')" \
                          --test_dataset=" 1_000 @ PointodysseyDatasets(split='train', ROOT='$dataset_root/PointOdyssey_proc/train', resolution=224, seed=777, depth_prior_name='moge')+ 1_000 @ SpringDatasets(split='train', ROOT='$dataset_root/spring_proc/train', resolution=224, seed=777, depth_prior_name='moge') + 1_000 @ SceneFlowDatasets(split='test', ROOT='$dataset_root/SceneFlow/', resolution=224, seed=777, depth_prior_name='moge')" \
                          --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
                          --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
                          --model="AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
                          --pretrained="ckpt/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
                          --lr=0.00005 --min_lr=1e-06 --warmup_epochs=5 --epochs=50 --batch_size=2 --accum_iter=4 \
                          --save_freq=1 --keep_freq=5 --eval_freq=1 --print_freq=10 --disable_cudnn_benchmark \
                          --output_dir="checkpoints/Align3R_512dpt_finedynamic"


echo "Press Enter to continue..."
read -p ""