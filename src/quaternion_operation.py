import numpy as np
import math
import autograd.numpy as np  # Autograd-compatible NumPy for automatic differentiation


def compute_exp_quat(q):
    """
    Compute the exponential of a quaternion.
    """
    s, v = q[0], q[1:]
    v_norm = np.linalg.norm(v)

    scalar_factor = np.exp(s)

    if v_norm > 0:
        vec_factor = v * (np.sin(v_norm) / v_norm)
    else:
        vec_factor = v

    exp_q = scalar_factor * (np.array([np.cos(v_norm)] + list(vec_factor)))
    return exp_q


def compute_exp_quat_mat(q):
    """
    Compute the exponential of a batch of quaternions.
    """
    vmags = np.linalg.norm(q[:, 1:], axis=1)
    scalar_exp = np.exp(q[:, 0])
    q_exp = np.zeros_like(q)
    q_exp[:, 0] = scalar_exp * np.cos(vmags)
    
    nz = vmags > 0  # Non-zero vector magnitudes
    if nz.any():
        v_norm = q[nz, 1:] / vmags[nz, np.newaxis]
        q_exp[nz, 1:] = scalar_exp[nz, np.newaxis] * np.sin(vmags[nz])[:, np.newaxis] * v_norm
    
    return q_exp


def compute_quat_inv_mat(q):
    """
    Compute the inverse of a batch of quaternions.
    """
    num = np.array([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]]).T
    den = np.expand_dims(np.square(np.linalg.norm(q, axis=1)), axis=1) + 1e-10  # Avoid division by zero
    return num / den


def compute_quat_prod(q, p):
    """
    Compute the Hamilton product of two quaternions.
    """
    q_s, q_v = q[0], q[1:]
    p_s, p_v = p[0], p[1:]

    scalar = q_s * p_s - np.dot(q_v, p_v)
    vector = q_s * np.array(p_v) + p_s * np.array(q_v) + np.cross(q_v, p_v)
    
    return np.array([scalar] + list(vector))


def compute_quat_prod_mat(q, p):
    """
    Compute the Hamilton product for batches of quaternions.
    """
    q_s, q_v = q[:, 0], q[:, 1:]
    p_s, p_v = p[:, 0], p[:, 1:]

    scalar = q_s * p_s - np.sum(q_v * p_v, axis=1)
    scalar = np.expand_dims(scalar, axis=1)
    
    vector = np.expand_dims(q_s, axis=1) * np.array(p_v) + \
        np.expand_dims(p_s, axis=1) * np.array(q_v) + np.cross(q_v, p_v)

    return np.hstack((scalar, vector))


def compute_log_quat(q):
    """
    Compute the logarithm of a quaternion.
    """
    s, v = q[0], q[1:]
    v_norm = np.linalg.norm(v)
    
    if v_norm > 0:
        v_factor = v * (np.arccos(s) / v_norm)
    else:
        v_factor = v
    
    log_q = np.array([0] + list(v_factor))
    return log_q


def compute_log_quat_mat(q):
    """
    Compute the logarithm of a batch of quaternions.
    """
    qv_mag = np.linalg.norm(q[:, 1:], axis=1) + 1e-10
    q_mag = np.linalg.norm(q, axis=1) + 1e-10
    
    first_col = np.log(q_mag)
    sec_to_four_col = np.expand_dims(np.arccos(q[:, 0] / q_mag), axis=1) * q[:, 1:] / np.expand_dims(qv_mag, axis=1)
    
    log_q = np.array([first_col, sec_to_four_col[:, 0], sec_to_four_col[:, 1], sec_to_four_col[:, 2]])
    return log_q


def compute_f_mat(q, tau, omega):
    """
    Compute quaternion state transition for a batch.
    """
    quat_omega = np.zeros_like(q)
    quat_omega[:, 1:] = tau * omega / 2
    exp_quat_omega = compute_exp_quat_mat(quat_omega)
    f_out = compute_quat_prod_mat(q, exp_quat_omega)
    
    return f_out


def compute_h_mat(q, GRAVITY):
    """
    Compute the measurement function for quaternion-based gravity estimation.
    """
    tmp = np.zeros_like(q)
    tmp[:, 3] = -GRAVITY
    
    return compute_quat_prod_mat(compute_quat_prod_mat(compute_quat_inv_mat(q), tmp), q)


def compute_f(q, tau, omega):
    """
    Compute quaternion state transition for a single quaternion.
    """
    quat_omega = np.insert(tau * omega / 2, 0, 0)
    exp_quat_omega = compute_exp_quat(quat_omega)
    f_out = compute_quat_prod(q, exp_quat_omega)
    
    return f_out
