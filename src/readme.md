# ECE276A: Project 1 - Orientation Tracking

## Overview
This project implements an orientation tracking algorithm using data from an Inertial Measurement Unit (IMU) and generates a panoramic image by stitching camera images from a rotating body. The project consists of multiple Python scripts for data loading, quaternion operations, orientation tracking, and visualization.

## File Descriptions

### Code Files
- **load_data.py**: Loads IMU, VICON, and camera data from the provided datasets.
- **quaternion_operation.py**: Implements quaternion operations, including multiplication, normalization, and conversion between representations.
- **orientation_tracking.py**: Implements the projected gradient descent algorithm to estimate orientation from IMU data.
- **rotplot.py**: Plots estimated versus ground truth roll, pitch, and yaw angles.
- **plotting.py**: Contains additional plotting utilities for data visualization.
- **panorama.py**: Implements the panorama generation algorithm using estimated orientations.

### Dependencies
The following Python libraries are required to run the project:
- `numpy`
- `scipy`
- `matplotlib`
- `transforms3d`
- `opencv-python`
- `jax`

Ensure all dependencies are installed before running the code. You can install them using:
```sh
pip install numpy scipy matplotlib transforms3d opencv-python jax
```

## Running the Code

### 1. Load and Preprocess Data
Run the following script to load the IMU, VICON, and camera data:
```sh
python load_data.py
```
This script reads the dataset and prepares it for further processing.

### 2. Compute Orientation Using IMU Data
To estimate the orientation using the projected gradient descent algorithm, run:
```sh
python orientation_tracking.py
```
This script computes and saves the estimated quaternion orientations.

### 3. Visualize Orientation Tracking
To visualize roll, pitch, and yaw angles against ground truth, run:
```sh
python rotplot.py
```
This generates plots comparing estimated and ground-truth orientations.

### 4. Generate Panorama
To stitch camera images into a panorama using estimated orientations, run:
```sh
python panorama.py
```
This will produce a stitched image based on the estimated orientations.

## Notes
- Ensure that the dataset files are placed in the appropriate directory before running the scripts.
- The first few seconds of IMU data should be used to calibrate sensor biases.
- If orientation estimation is inaccurate, VICON ground-truth data can be used instead.

## Contact
For any questions or issues, please refer to the project documentation or contact the course staff.