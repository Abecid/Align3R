from PIL import Image
from tqdm import tqdm
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib
import argparse
from transformers import pipeline
import torch

import depth_pro
from third_party.MoGe.moge.model import MoGeModel

def find_images(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('rgb.jpg', 'rgb.png')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Load model and preprocessing transform
parser = argparse.ArgumentParser('ml infer', add_help=False)
parser.add_argument('--a', default=0,
                        type=int)
parser.add_argument('--b', default=1500,
                        type=int)
parser.add_argument('--dataset_name', default=None,
                        type=str)
parser.add_argument('--model_name', default="moge",
                        type=str)
parser.add_argument('--data_path', default="/scratch/partial_datasets/align3r/data",
                        type=str)
args = parser.parse_args()
# print(args.a)
model, transform = depth_pro.create_model_and_transforms(device='cuda')
model.eval()
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf",device='cuda')

if args.dataset_name == "Tartanair":
  dir = f'{args.data_path}/Tartanair_proc/'
elif args.dataset_name == "spring":
  dir = f'{args.data_path}/spring_proc/train/'
elif args.dataset_name == "SceneFlow":
  dir = f'{args.data_path}/SceneFlow/'
elif args.dataset_name == "Vkitti":
  dir = f'{args.data_path}/vkitti_2.0.3_proc/' 
elif args.dataset_name == "PointOdyssey":
  dir = f'{args.data_path}/PointOdyssey_proc/' 

image_paths = find_images(dir)
if args.model_name == "depthpro":
  for image_path in tqdm(sorted(image_paths)[int(args.a):int(args.b)]):
    # depthanything v2
    image = Image.open(image_path)
    depth = pipe(image)["predicted_depth"].numpy()
    #depth = prediction["depth"].cpu()  # Depth in [m].
    metadata = np.load(image_path.replace('_rgb.jpg', '_metadata.npz'))
    intrinsics = np.float32(metadata['camera_intrinsics'])
    focallength_px = intrinsics[0][0]
    np.savez_compressed(image_path[:-4]+'_pred_depth_depthanything', depth=depth,focallength_px=focallength_px)
    # depthpro
    image, _, f_px = depth_pro.load_rgb(image_path)
    image = transform(image)
    # Run inference.
    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"].cpu()  # Depth in [m].
    np.savez_compressed(image_path[:-4]+'_pred_depth_depthpro', depth=depth, focallength_px=prediction["focallength_px"].cpu())  
elif args.model_name == "moge":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
  for image_path in tqdm(sorted(image_paths)[int(args.a):int(args.b)]):
    path_moge = image_path.replace('.png','_pred_depth_moge.npz').replace('.jpg','_pred_depth_moge.npz')

    metadata = np.load(image_path.replace('_rgb.jpg', '_metadata.npz'))
    intrinsics = np.float32(metadata['camera_intrinsics'])
    focallength_px = intrinsics[0][0]

    # Read the input image and convert to tensor (3, H, W) and normalize to [0, 1]
    input_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

    # Infer 
    output = model.infer(input_image)
    depth = output["depth"].cpu().numpy()
    mask = output["mask"].cpu().numpy()
    intrinsics = output["intrinsics"].cpu().numpy() # (3, 3)
    focal_length_x, focal_length_y = intrinsics[0, 0], intrinsics[1, 1]
    
    np.savez_compressed(path_moge, depth=depth, mask=mask, focallength_px=focallength_px)
