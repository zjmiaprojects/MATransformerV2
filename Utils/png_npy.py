import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Define the transformation to resize the image
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize the image to 512x512
    transforms.ToTensor()  # Convert image to PyTorch tensor
])


# Function to convert a PNG image to a .npy file
def png_to_npy(png_path, npy_path):
    # Open the image file
    img = Image.open(png_path)
    print("img:",img.size)

    # Apply the transformations
    img = transform(img)

    # Convert the tensor to a numpy array
    img_np = img.numpy()
    print("img_np:", img_np.shape)
    image_np = img_np / 255.0
    # Save the numpy array as a .npy file
    np.save(npy_path, img_np)


# # Example usage:
# png_file_path = "/opt/data/private/dataset/piccolo/train/polyps/001_VP1_frame0000.png"  # Input PNG file path
# npy_file_path = "/opt/data/private/dataset/piccolo/train/001_VP1_frame0000.npy"  # Output .npy file path
#
# png_to_npy(png_file_path, npy_file_path)


import os
input_pth = '/opt/data/private/dataset/piccolo/test/polyps/'
output_pth = '/opt/data/private/dataset/piccolo/test_512/img/'
data_list = os.listdir(input_pth)
print(data_list)

for i in range(len(data_list)):
        print(data_list[i])
        name=data_list[i].split(".")
        print(name[0])

        input_tiff = os.path.join(input_pth, data_list[i])
        output_npy = os.path.join(output_pth, name[0]+'.npy')
        png_to_npy(input_tiff, output_npy)
