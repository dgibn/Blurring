import cv2
import os
import numpy as np
from tqdm import tqdm

src_dir = '/home/divs/Desktop/CSRE/Blur/sharp'


images = os.listdir(src_dir)

dst_dir = '/home/divs/Desktop/CSRE/Blur/Gaussian'

for i, image in tqdm(enumerate(images),total=len(images)):
    
    img=cv2.imread(f"{src_dir}/{images[i]}",cv2.IMREAD_COLOR)

    blur =cv2.GaussianBlur(img,(29,29),0)

    cv2.imwrite(f"{dst_dir}/{images[i]}",blur)

