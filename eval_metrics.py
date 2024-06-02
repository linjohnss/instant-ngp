import os
import csv
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import torch
import argparse

# Initialize LPIPS model
loss_fn = lpips.LPIPS(net='alex')

# Function to read images
def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Function to calculate PSNR
def calculate_psnr(img1, img2):
    return psnr(img1, img2)

# Function to calculate SSIM
def calculate_ssim(img1, img2):
    return ssim(img1, img2, channel_axis=-1, data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

# Function to calculate LPIPS
def calculate_lpips(img1, img2):
    img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return loss_fn(img1, img2).item()

def main(args):
    # List images in the folders
    gt_images = sorted(os.listdir(args.gt_folder))
    pred_images = sorted(os.listdir(args.pred_folder))

    # Initialize lists to store metrics
    psnr_values = []
    ssim_values = []
    lpips_values = []

    # Calculate metrics for each image pair based on image name
    for gt_image in gt_images:
        if gt_image in pred_images:
            gt_image_path = os.path.join(args.gt_folder, gt_image)
            pred_image_path = os.path.join(args.pred_folder, gt_image)

            gt_img = read_image(gt_image_path)
            pred_img = read_image(pred_image_path)

            psnr_values.append(calculate_psnr(gt_img, pred_img))
            ssim_values.append(calculate_ssim(gt_img, pred_img))
            lpips_values.append(calculate_lpips(gt_img, pred_img))

    # Calculate average metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_lpips = np.mean(lpips_values)

    # Save average metrics to CSV
    csv_file = args.csv_file
    csv_columns = ['Average_PSNR', 'Average_SSIM', 'Average_LPIPS']
    csv_data = [{
        'Average_PSNR': avg_psnr,
        'Average_SSIM': avg_ssim,
        'Average_LPIPS': avg_lpips
    }]

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerow(csv_data[0])

    print(f"Average metrics saved to {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate PSNR, SSIM, and LPIPS metrics for images in two folders and save the average metrics.")
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the output CSV file')
    parser.add_argument('--gt_folder', type=str, required=True, help='Path to the ground truth images folder')
    parser.add_argument('--pred_folder', type=str, required=True, help='Path to the predicted images folder')
    
    args = parser.parse_args()
    main(args)
