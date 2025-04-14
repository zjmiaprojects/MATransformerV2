from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
img=[]
img1=Image.open('../Utils/feature_map_output/data/MATransformerV2/kMeans_60_x_code/PSNR_4class/0_final.jpg')
img2=Image.open('../Utils/feature_map_output/data/MATransformerV2/kMeans_60_x_code/PSNR_4class/0_iter9.jpg')
img1 = np.array(img1)
img2 = np.array(img2)
img.append(img1)
img.append(img2)

# print(img[1])
print(type(img2))

if __name__ == "__main__":
	# If the input is a multichannel (color) image, set multichannel=True.
    print(ssim(img1, img2, multichannel=True))
