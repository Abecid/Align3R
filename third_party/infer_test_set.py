from PIL import Image
from tqdm import tqdm
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
import cv2

import torch
from transformers import pipeline

from third_party.MoGe.moge.model import MoGeModel

def find_images(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
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

if args.model_name == "depthpro":
  import depth_pro
  model, transform = depth_pro.create_model_and_transforms(device='cuda')
  model.eval()
  pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf",device='cuda')

if args.dataset_name == "bonn":
  dir = f'{args.data_path}/bonn/rgbd_bonn_dataset/'
elif args.dataset_name == "davis":
  dir = f'{args.data_path}/davis/DAVIS/JPEGImages/480p/'
elif args.dataset_name == "sintel":
  dir = f'{args.data_path}/MPI-Sintel/MPI-Sintel-training_images/training/final/'
elif args.dataset_name == "tum":
  dir = f'{args.data_path}/tum/'

if args.model_name == "moge":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)

for scene in tqdm(sorted(os.listdir(dir))):
  data_dir = dir + scene
  if os.path.isdir(data_dir):
    if args.dataset_name == "bonn":
      data_dir = data_dir + '/rgb_110'
    elif args.dataset_name == "tum":
      data_dir = data_dir + '/rgb_50'
    for image_path in tqdm(sorted(os.listdir(data_dir))[int(args.a):int(args.b)]):
      #print(image_path)
      if image_path.split('.')[-1]=='jpg' or image_path.split('.')[-1]=='png': 
        if args.dataset_name == "bonn":
          if not os.path.exists(data_dir.replace('rgb_110', 'rgb_110_depth_prediction_depthanything')):
            os.makedirs(data_dir.replace('rgb_110', 'rgb_110_depth_prediction_depthanything'))
          if not os.path.exists(data_dir.replace('rgb_110', 'rgb_110_depth_prediction_depthpro')):
            os.makedirs(data_dir.replace('rgb_110', 'rgb_110_depth_prediction_depthpro'))
          path_depthanything = os.path.join(data_dir, image_path).replace('rgb_110', 'rgb_110_depth_prediction_depthanything').replace('.jpg', '.npz').replace('.png', '.npz')
          path_depthpro = os.path.join(data_dir, image_path).replace('rgb_110', 'rgb_110_depth_prediction_depthpro').replace('.jpg', '.npz').replace('.png', '.npz')
          path_moge = os.path.join(data_dir, image_path).replace('rgb_110', 'rgb_110_depth_prediction_moge').replace('.jpg', '.npz').replace('.png', '.npz')
        elif args.dataset_name == "tum":
          if not os.path.exists(data_dir.replace('rgb_50', 'rgb_50_depth_prediction_depthanything')):
            os.makedirs(data_dir.replace('rgb_50', 'rgb_50_depth_prediction_depthanything'))
          if not os.path.exists(data_dir.replace('rgb_50', 'rgb_50_depth_prediction_depthpro')):
            os.makedirs(data_dir.replace('rgb_50', 'rgb_50_depth_prediction_depthpro'))
          path_depthanything = os.path.join(data_dir, image_path).replace('rgb_50', 'rgb_50_depth_prediction_depthanything').replace('.jpg', '.npz').replace('.png', '.npz')
          path_depthpro = os.path.join(data_dir, image_path).replace('rgb_50', 'rgb_50_depth_prediction_depthpro').replace('.jpg', '.npz').replace('.png', '.npz')
          path_moge = os.path.join(data_dir, image_path).replace('rgb_50', 'rgb_50_depth_prediction_moge').replace('.jpg', '.npz').replace('.png', '.npz')
        elif args.dataset_name == "sintel":
          if not os.path.exists(data_dir.replace('final', 'depth_prediction_depthanything')):
            os.makedirs(data_dir.replace('final', 'depth_prediction_depthanything'))
          if not os.path.exists(data_dir.replace('final', 'depth_prediction_depthpro')):
            os.makedirs(data_dir.replace('final', 'depth_prediction_depthpro'))
          path_depthanything = os.path.join(data_dir, image_path).replace('final', 'depth_prediction_depthanything').replace('.jpg', '.npz').replace('.png', '.npz')
          path_depthpro = os.path.join(data_dir, image_path).replace('final', 'depth_prediction_depthpro').replace('.jpg', '.npz').replace('.png', '.npz')
          path_moge = os.path.join(data_dir, image_path).replace('final', 'depth_prediction_moge').replace('.jpg', '.npz').replace('.png', '.npz')
        elif args.dataset_name == "davis":
          if not os.path.exists(data_dir.replace('JPEGImages', 'depth_prediction_depthanything')):
              os.makedirs(data_dir.replace('JPEGImages', 'depth_prediction_depthanything'))
          if not os.path.exists(data_dir.replace('JPEGImages', 'depth_prediction_depthpro')):
            os.makedirs(data_dir.replace('JPEGImages', 'depth_prediction_depthpro'))
          path_depthanything = os.path.join(data_dir, image_path).replace('JPEGImages', 'depth_prediction_depthanything').replace('.jpg', '.npz').replace('.png', '.npz')
          path_depthpro = os.path.join(data_dir, image_path).replace('JPEGImages', 'depth_prediction_depthpro').replace('.jpg', '.npz').replace('.png', '.npz')
          path_moge = os.path.join(data_dir, image_path).replace('JPEGImages', 'depth_prediction_moge').replace('.jpg', '.npz').replace('.png', '.npz')

        if args.model_name == "depthpro":
          # depthanything v2
          image = Image.open(os.path.join(data_dir, image_path))
          depth = pipe(image)["predicted_depth"].numpy()
          #depth = prediction["depth"].cpu()  # Depth in [m].
          np.savez_compressed(path_depthanything, depth=depth)  
          # depthpro
          image, _, f_px = depth_pro.load_rgb(os.path.join(data_dir, image_path))
          image = transform(image)
          # Run inference.
          prediction = model.infer(image, f_px=f_px)
          depth = prediction["depth"].cpu()  # Depth in [m].
          np.savez_compressed(path_depthpro, depth=depth, focallength_px=prediction["focallength_px"].cpu())  
        elif args.model_name == "moge":
          input_image = cv2.cvtColor(cv2.imread(os.path.join(data_dir, image_path)), cv2.COLOR_BGR2RGB)
          input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

          # Infer 
          output = model.infer(input_image)
          depth = output["depth"].cpu().numpy()
          mask = output["mask"].cpu().numpy()
          intrinsics = output["intrinsics"].cpu().numpy() # (3, 3)
          focal_length_x, focal_length_y = intrinsics[0, 0], intrinsics[1, 1]
          
          np.savez_compressed(path_moge, depth=depth, mask=mask, focallength_px=focal_length_x)