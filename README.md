# IMU Orientation Panorama

## 📌 Project Overview
This project implements a robust method for tracking the 3D orientation of a rotating body using IMU (Inertial Measurement Unit) data and projected gradient descent optimization. The estimated orientations are utilized to create coherent panoramic images by stitching sequential camera frames.

## 🔍 Technical Details

### 1️⃣ IMU Calibration
- **Accelerometer and Gyroscope Bias Calibration** performed using static segments of IMU data.
- **Sensor Synchronization** using timestamp interpolation.
- **Unit Conversion** of IMU measurements (degrees/sec to radians/sec).

### 2️⃣ Orientation Estimation (Projected Gradient Descent)
- **Quaternion Motion Model** for orientation evolution.
- **Observation Model** aligns measured acceleration with gravitational acceleration.
- **Optimization Problem** formulated and solved via projected gradient descent ensuring quaternion unit-norm constraint.
- **Convergence Analysis** validated through iterative cost function evaluation.

### 3️⃣ Panorama Image Stitching
- **Spherical Coordinate Mapping** transforming camera images based on estimated orientations.
- **Image Projection and Blending** employing rotation matrices and equirectangular projection.

## 📂 Repository Structure
```
.
├── docs/
│   ├── accelerometer_datasheet_adxl335.pdf
│   ├── gyroscope_pitch_roll_datasheet_lpr530al.pdf
│   ├── gyroscope_yaw_datasheet_ly530alh.pdf
│   ├── IMU_reference.pdf
│   └── IMUandCamSensors.jpg
├── report/
│   └── ECE276A_Project1_Report.pdf
├── result_plots/
│   ├── dataset_1/
│   ├── dataset_2/
│   ├── ... (other datasets)
│   └── dataset_11/
└── src/
    ├── load_data.py
    ├── orientation_tracking.py
    ├── panorama.py
    ├── plotting.py
    ├── quaternion_operation.py
    ├── rotplot.py
    └── README.md
```

## 📈 Results & Evaluation
- Estimated roll, pitch, and yaw angles closely match ground truth (mean absolute errors under 0.5°).
- Effective panorama stitching verified through coherent visual reconstruction, despite minor yaw drift in prolonged sequences.

## 🛠️ Technologies
- **Languages & Libraries**: Python, NumPy, JAX, transforms3d
- **Data Handling & Visualization**: Matplotlib, Pandas

## 🎯 Future Improvements
- Integration of sensor fusion techniques to mitigate drift.
- Real-time filtering methods (EKF, complementary filter).
- GPU-based parallel processing for optimization and panorama rendering.
- Adaptive learning rate strategies (Adam, RMSprop).

## 📚 Documentation & References
- Detailed project report available in [`report/ECE276A_Project1_Report.pdf`](report/ECE276A_Project1_Report.pdf).
- Sensor datasheets and references located under [`docs/`](docs/).

## 👤 Author
**Brian (Shou Yu) Wang** – [BrianSY541](https://github.com/BrianSY541)

📧 Contact: briansywang541@gmail.com | www.linkedin.com/in/sywang541
