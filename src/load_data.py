import pickle
import sys
import time
import numpy as np

def tic():
    return time.time()

def toc(tstart, nm=""):
    print('%s took: %s sec.\n' % (nm, (time.time() - tstart)))

def read_data(fname):
    d = []
    try:
        with open(fname, 'rb') as f:
            if sys.version_info[0] < 3:
                d = pickle.load(f)
            else:
                d = pickle.load(f, encoding='latin1')  # needed for python 3
    except FileNotFoundError:
        print(f"No such file: {fname}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return d

def load_dataset(dataset_idx):
    dataset = str(dataset_idx)
    cfile = "../data/cam/cam" + dataset + ".p"
    ifile = "../data/imu/imuRaw" + dataset + ".p"
    vfile = "../data/vicon/viconRot" + dataset + ".p"

    ts = tic()

    camd = read_data(cfile)
    imud = read_data(ifile)
    vicd = read_data(vfile)

    if len(camd) != 0:
        cam_arr, cam_ts = np.transpose(
            camd['cam'], (3, 0, 1, 2)), camd['ts'].T
    else:
        cam_arr, cam_ts = None, None

    imu_arr, imu_ts = np.array(
        imud['vals'].T, dtype=np.float32), imud['ts'].T
    # (Ax, Ay, Az, Wx, Wy, Wz)
    imu_arr[:, [5, 3, 4]] = imu_arr[:, [3, 4, 5]]

    if len(vicd) != 0:
        vic_arr, vic_ts = np.transpose(
            vicd['rots'], (2, 0, 1)), vicd['ts'].T
    else:
        vic_arr, vic_ts = None, None

    toc(ts, "Data import")

    return cam_arr, cam_ts, imu_arr, imu_ts, vic_arr, vic_ts
