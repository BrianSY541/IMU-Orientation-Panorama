import numpy as np
import cv2
import glob
import os
from scipy.spatial.transform import Rotation as R
from load_data import load_dataset  # Ensure dataset loading function is included

# Load images from the given directory and sort them by timestamp
def load_images(image_folder):
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.png')))
    images = [cv2.imread(img) for img in image_files]
    return images, image_files

# Parse orientation data, converting quaternions to rotation matrices
def parse_orientation(orientation_file):
    data = np.loadtxt(orientation_file, delimiter=',')
    timestamps, quaternions = data[:, 0], data[:, 1:]
    rotation_matrices = [R.from_quat(q).as_matrix() for q in quaternions]
    return timestamps, rotation_matrices

# Compute homography transformations from rotation matrices
def compute_homography(rotations):
    homographies = []
    reference_rotation = rotations[0]
    
    for R_current in rotations:
        H = np.dot(reference_rotation.T, R_current)
        homographies.append(H)
    
    return homographies

# Stitch images together using computed homographies
def stitch_images(images, homographies):
    panorama = None
    h, w, _ = images[0].shape
    
    for img, H in zip(images, homographies):
        warped = cv2.warpPerspective(img, H, (w * 2, h * 2))
        if panorama is None:
            panorama = warped
        else:
            panorama = np.maximum(panorama, warped)
    
    return panorama

# Main function to execute the full panorama generation pipeline
def generate_panorama(image_dir, orientation_file, output_file):
    dataset = load_dataset(image_dir)  # Utilize dataset loading function
    images, _ = load_images(image_dir)
    timestamps, rotations = parse_orientation(orientation_file)
    homographies = compute_homography(rotations)
    panorama = stitch_images(images, homographies)
    cv2.imwrite(output_file, panorama)
    print("Panorama saved at:", output_file)

# Example execution
if __name__ == "__main__":
    image_directory = "images"
    orientation_data = "orientation.csv"
    output_path = "output_panorama.png"
    generate_panorama(image_directory, orientation_data, output_path)
