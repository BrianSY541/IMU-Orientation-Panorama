import os
import time

from transforms3d.euler import mat2euler
from transforms3d.euler import quat2euler
import autograd
import autograd.numpy as np

from quaternion_operation import *
from plotting import *
from load_data import load_dataset

# Define gravity constant for sensor calibration
GRAVITY = -1.0

def IMU_transform(imu_data, calibration_frames):
    """
    Calibrate and transform raw IMU data to physical units.
    
    Args:
        imu_data: Raw IMU measurements (accelerometer and gyroscope)
        calibration_frames: Number of initial frames used for bias estimation
        
    Returns:
        imu_cal: Calibrated IMU data in physical units (m/s^2 for accel, rad/s for gyro)
    """
    # Scale factors for converting ADC values to physical units
    scale_factor_a = 3300 / 1023 / 300  # Accelerometer scale (m/s^2)
    scale_factor_w = 3300 / 1023 / 3.33 * np.pi / 180  # Gyroscope scale (rad/s)
    
    # Flip Ax and Ay directions due to sensor mounting
    imu_data[:, 0:2] *= -1.0

    # Calculate accelerometer bias using initial frames
    bias_a = np.mean(imu_data[0:calibration_frames, 0:3], axis=0)
    bias_g = np.abs(GRAVITY) / scale_factor_a
    # Remove gravity component from z-axis bias
    bias_a[2] = bias_a[2] - bias_g

    # Calculate gyroscope bias
    bias_w = np.mean(imu_data[0:calibration_frames, 3:6], axis=0)

    # Apply calibration and scaling
    imu_cal = np.zeros_like(imu_data)
    imu_cal[:, 0:3] = (imu_data[:, 0:3] - bias_a) * scale_factor_a
    imu_cal[:, 3:6] = (imu_data[:, 3:6] - bias_w) * scale_factor_w

    return imu_cal

def vicon_rotation_to_euler(vicon_data):
    """
    Convert Vicon rotation matrices to Euler angles in degrees.
    
    Args:
        vicon_data: Array of 3x3 rotation matrices from Vicon
        
    Returns:
        vicon_degrees: Array of Euler angles in degrees (roll, pitch, yaw)
    """
    num_timestamps = vicon_data.shape[0]
    vicon_degrees = np.zeros((num_timestamps, 3), dtype=np.float32)

    for t in range(num_timestamps):
        euler_angles_rad = mat2euler(vicon_data[t], axes='sxyz')
        vicon_degrees[t, :] = np.degrees(euler_angles_rad)

    return vicon_degrees

def IMU_to_euler(imu_data, timestamps):
    """
    Convert IMU measurements to orientation angles in degrees using quaternion integration.
    
    Args:
        imu_data: Calibrated IMU measurements
        timestamps: Corresponding time values
        
    Returns:
        imu_degrees: Array of estimated Euler angles in degrees (roll, pitch, yaw)
    """
    num_timestamps = imu_data.shape[0]
    quaternion_pred = np.zeros((num_timestamps, 4), dtype=np.float32)
    imu_degrees = np.zeros((num_timestamps, 3), dtype=np.float32)

    # Initialize with identity quaternion
    quaternion_pred[0] = np.array([1, 0, 0, 0], dtype=np.float32)
    euler_angles_rad = quat2euler(quaternion_pred[0], axes='sxyz')
    imu_degrees[0, :] = np.degrees(euler_angles_rad)

    # Integrate angular velocity to get orientations
    for t in range(num_timestamps - 1):
        time_delta = timestamps[t+1] - timestamps[t]
        angular_velocity = imu_data[t, 3:6]
        quaternion_pred[t+1] = compute_f(quaternion_pred[t], time_delta, angular_velocity)
        euler_angles_rad = quat2euler(quaternion_pred[t+1], axes='sxyz')
        imu_degrees[t+1, :] = np.degrees(euler_angles_rad)

    return imu_degrees

def compute_orientation_estimation_cost(quaternion_array):
    """
    Compute the cost function for orientation estimation using matrix operations.
    
    Args:
        quaternion_array: Array of quaternions representing orientations
        
    Returns:
        cost: Scalar cost value combining prediction and measurement terms
    """
    q0 = np.array([1, 0, 0, 0], dtype=np.float32)
    num_timestamps = quaternion_array.shape[0]

    # Compute inverse quaternions and gravity vectors
    q_inv_mat = compute_quat_inv_mat(quaternion_array)
    h_mat = compute_h_mat(quaternion_array, GRAVITY)[:, 1:]

    # Stack initial quaternion with prediction array
    quaternion_stack = np.vstack([q0, quaternion_array[0:-1]])

    # Compute time deltas and prepare angular velocities
    time_deltas = imu_ts[1:] - imu_ts[0:-1]
    angular_velocities = imu_arr_cal[:-1, 3:6]

    # Compute predicted orientations
    predicted_quaternions = compute_f_mat(quaternion_stack, time_deltas, angular_velocities)
    
    # Add small noise to avoid numerical issues
    noise = np.random.rand(quaternion_array.shape[0], quaternion_array.shape[1]) * 1e-10
    quaternion_diff = 2 * compute_log_quat_mat(
        compute_quat_prod_mat(q_inv_mat, predicted_quaternions) + noise)

    # Compute prediction and measurement error terms
    prediction_error = np.sum(np.square(np.linalg.norm(quaternion_diff, axis=1)))
    measurement_error = np.sum(np.square(np.linalg.norm(
        imu_arr_cal[1:, 0:3] - h_mat, axis=1)))

    # Combine error terms
    cost = 0.5 * (prediction_error + measurement_error)
    return cost

def PGD_optimization(imu_data_cal, timestamps, dataset_idx, iterations=100, learning_rate=1e-2):
    """
    Optimize orientation estimation using Projected Gradient Descent.
    
    Args:
        imu_data_cal: Calibrated IMU measurements
        timestamps: Time values for measurements
        dataset_idx: Dataset identifier for plotting
        iterations: Number of optimization iterations
        learning_rate: Learning rate for gradient descent
        
    Returns:
        optimized_quaternions: Array of optimized quaternion orientations
    """
    # Initialize with identity quaternion
    q0 = np.array([1, 0, 0, 0], dtype=np.float32)
    num_timestamps = imu_data_cal.shape[0]
    num_states = num_timestamps - 1

    # Initialize quaternion array using simple integration
    optimized_quaternions = np.zeros((num_timestamps-1, 4), dtype=np.float32)
    for t in range(num_timestamps - 1):
        time_delta = timestamps[t+1] - timestamps[t]
        angular_velocity = imu_data_cal[t, 3:6]
        if t == 0:
            optimized_quaternions[t] = compute_f(q0, time_delta, angular_velocity)
        else:
            optimized_quaternions[t] = compute_f(
                optimized_quaternions[t-1], time_delta, angular_velocity)

    # Store cost history for plotting
    cost_history = [compute_orientation_estimation_cost(optimized_quaternions)]
    gradient_function = autograd.jacobian(compute_orientation_estimation_cost)

    # Optimization loop
    for _ in range(iterations):
        # Update quaternions using gradient descent
        optimized_quaternions = optimized_quaternions - learning_rate * gradient_function(optimized_quaternions)
        # Normalize quaternions
        optimized_quaternions = optimized_quaternions / \
            np.reshape(np.linalg.norm(optimized_quaternions, axis=1), (num_states, 1))
        cost_history.append(compute_orientation_estimation_cost(optimized_quaternions))

    # Plot optimization progress
    plt_cost(cost_history, dataset_idx)
    return optimized_quaternions

if __name__ == "__main__":
    # Process datasets
    for dataset_idx in range(1, 2):
        # Create output directories
        os.makedirs("../picture/dataset_" + str(dataset_idx), exist_ok=True)

        # Load dataset
        cam_arr, cam_ts, imu_arr, imu_ts, vic_arr, vic_ts = load_dataset(dataset_idx)
        num_imu_ts = imu_ts.shape[0]

        # Transform and calibrate IMU data
        imu_arr_cal = IMU_transform(imu_arr, 400)
        
        # Convert to Euler angles
        imu_deg_arr = IMU_to_euler(imu_arr_cal, imu_ts)
        
        # Plot results (with or without Vicon ground truth)
        if dataset_idx <= 9:
            vic_deg_arr = vicon_rotation_to_euler(vic_arr)
            plt_rpy(vic_deg_arr, vic_ts, imu_deg_arr, imu_ts, dataset_idx)
        else:
            plot_rpy_no_vicon(imu_deg_arr, imu_ts, dataset_idx)

        # Optimize orientation estimation
        optimized_quaternions = PGD_optimization(
            imu_arr_cal, imu_ts, dataset_idx, iterations=200, learning_rate=5e-3)

        # Convert optimized quaternions to Euler angles
        q0 = np.array([1, 0, 0, 0], dtype=np.float32)
        optimized_quaternions = np.vstack([q0, optimized_quaternions])

        # Save optimization results
        save_dir = "../data/optim/dataset_" + str(dataset_idx)
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/opt_quat_arr.npy", optimized_quaternions)

        # Convert to degrees and plot final results
        optimized_degrees = np.zeros((num_imu_ts, 3), dtype=np.float32)
        for t in range(num_imu_ts):
            euler_angles_rad = quat2euler(optimized_quaternions[t], axes='sxyz')
            optimized_degrees[t, :] = np.degrees(euler_angles_rad)

        if dataset_idx <= 9:
            plt_rpy(vic_deg_arr, vic_ts, optimized_degrees,
                   imu_ts, dataset_idx, optim=True)
        else:
            plot_rpy_no_vicon(optimized_degrees, imu_ts, dataset_idx, optim=True)