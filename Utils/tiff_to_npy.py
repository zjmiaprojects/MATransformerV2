import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import tifffile as tiff
import numpy as np
import torchvision.transforms as transforms
import torch


def tiff_to_npy(input_path, output_path, target_size=(1, 512, 512)):
    # 使用 tifffile 库打开 TIFF 文件
    image = tiff.imread(input_path)

    # 如果图像是灰度图像或单通道图像，扩展为 3 通道
    # if image.ndim == 2:
    #     image = np.stack([image] * 3, axis=-1)  # 扩展为 (height, width, 3)

    # 将图像转换为 PIL 图像
    image = Image.fromarray(image)
    print(image.size)

    # Resize the image to 512x512
    resized_image = image.resize((512, 512), Image.LANCZOS)

    # Convert the image to a numpy array
    image_np = np.array(resized_image)

    # If needed, convert the data type to float32 for compatibility with PyTorch
    image_np = image_np.astype(np.float32)

    # Optionally, normalize the image data (depending on your use case)
    image_np = image_np / 255.0  # normalize to [0, 1] range
    print("image_np:",image_np.shape)
    # Save the numpy array as a .npy file
    np.save(output_npy, image_np)
    print(f"Saved the file at {output_path} with shape {image_np.shape}")



# # 使用示例D:\pythonProject\unet\ClinicDB\CVC-ClinicDB\Original
# input_tiff = '/opt/data/private/dataset/piccolo/train/masks/001_VP1_frame0000_Corrected.tif'  # 替换为你的 TIFF 文件路径
# output_npy = '/opt/data/private/dataset/piccolo/train/001_VP1_frame0000_Corrected.npy'  # 替换为你想要保存的 .npy 文件路径
#
# tiff_to_npy(input_tiff, output_npy)

import os
input_pth = '/opt/data/private/dataset/piccolo/train/masks/'
output_pth = '/opt/data/private/dataset/piccolo/train_512/lab/'
data_list = os.listdir(input_pth)
print(data_list)

for i in range(len(data_list)):
        print(data_list[i])
        name=data_list[i].split(".")
        print(name[0])

        input_tiff = os.path.join(input_pth, data_list[i])
        output_npy = os.path.join(output_pth, name[0]+'.npy')
        tiff_to_npy(input_tiff, output_npy)

