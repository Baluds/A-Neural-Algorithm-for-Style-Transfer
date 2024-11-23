from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np

# Load the images (they should be the same size and in grayscale or RGB format)
originalContent = cv2.imread('/Users/balachandrads/Desktop/UMass/Fall 2024/602/Project/input/portrait.jpg')
originalStyle = cv2.imread('/Users/balachandrads/Desktop/UMass/Fall 2024/602/Project/input/escher.jpg')
stylized = cv2.imread('/Users/balachandrads/Desktop/UMass/Fall 2024/602/Project/Layers_combinations/combination_6_with_blank_settings/escher.png')

originalContent = cv2.resize(originalContent, (stylized.shape[1], stylized.shape[0]))
originalStyle = cv2.resize(originalStyle, (stylized.shape[1], stylized.shape[0]))

# Convert images to grayscale (optional if you want SSIM on a single channel)
original_gray = cv2.cvtColor(originalContent, cv2.COLOR_BGR2GRAY)
stylized_gray = cv2.cvtColor(stylized, cv2.COLOR_BGR2GRAY)
originalStyle_gray = cv2.cvtColor(originalStyle, cv2.COLOR_BGR2GRAY)


ssim_value_content = ssim(original_gray, stylized_gray)
ssim_value_style = ssim(originalStyle_gray, stylized_gray)


print(f"SSIM with Content Image: {ssim_value_content}")
print(f"SSIM with Style Image: {ssim_value_style}")


ssim_r_c = ssim(originalContent[:, :, 0], stylized[:, :, 0]) 
ssim_g_c = ssim(originalContent[:, :, 1], stylized[:, :, 1]) 
ssim_b_c = ssim(originalContent[:, :, 2], stylized[:, :, 2]) 

ssim_rgb_c = (ssim_r_c + ssim_g_c + ssim_b_c) / 3
print(f"SSIM with Content Image(RGB): {ssim_rgb_c}")


ssim_r_s = ssim(originalStyle[:, :, 0], stylized[:, :, 0]) 
ssim_g_s = ssim(originalStyle[:, :, 1], stylized[:, :, 1]) 
ssim_b_s = ssim(originalStyle[:, :, 2], stylized[:, :, 2]) 

ssim_rgb_s = (ssim_r_s + ssim_g_s + ssim_b_s) / 3
print(f"SSIM with Style Image(RGB): {ssim_rgb_s}")

