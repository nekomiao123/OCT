import math
import torch
import torchvision
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import cv2
import albumentations as A
from PIL import Image
from torch.optim import lr_scheduler
import torchvision.transforms as transforms


def svdna(k, src_path, target_path, histo_matching_degree=0.5, image_size=256):
    
    target_img = Image.open(target_path) 
    src_img = Image.open(src_path)  

    resized_target=np.array(target_img.resize((image_size,image_size), Image.Resampling.NEAREST))
    resized_src=np.array(src_img.resize((image_size,image_size), Image.Resampling.NEAREST))

    u_target,s_target,vh_target=np.linalg.svd(resized_target,full_matrices=False)
    u_source,s_source,vh_source=np.linalg.svd(resized_src,full_matrices=False)

    thresholded_singular_target=s_target
    thresholded_singular_target[0:k]=0

    thresholded_singular_source=s_source
    thresholded_singular_source[k:]=0

    target_style=np.array([np.dot(u_target, np.dot(np.diag(thresholded_singular_target), vh_target))])

    content_src=np.array([np.dot(u_source, np.dot(np.diag(thresholded_singular_source), vh_source))])
    content_trgt=resized_target-target_style

    noise_adapted_im=content_src+target_style

    noise_adapted_im_clipped=np.squeeze(noise_adapted_im).clip(0,255).astype(np.uint8)
  
    transformHist = A.Compose([
        A.HistogramMatching([target_path], blend_ratio=(histo_matching_degree, histo_matching_degree), read_fn=readIm, p=1)
    ])

    image = np.array(Image.open(src_path).resize((image_size,image_size)))

    transformed = transformHist(image=noise_adapted_im_clipped)
    svdna_im = transformed["image"]

    svdna_im = Image.fromarray(svdna_im).convert('RGB')
    
    return svdna_im

def readIm(imagepath):
  image = cv2.imread(str(imagepath),0)
  return image