"""
Forward simulation script
Generate synthetic groundwater dataset for HT-PINN inversion
Journal standard: Water Resources Research
Author: Guardian-ZhangHan
Version: 1.0.0
"""
import logging
import os
import time
import numpy as np
import scipy.io as sio

from utils import set_seed, ensure_dir, setup_logging

def main():
    setup_logging('../logs/forward.log')
    logging.info("="*60)
    logging.info("Starting forward simulation")
    logging.info("Generating synthetic dataset HT_synthetic.mat")
    logging.info("="*60)

    set_seed(42)

    # Auto create all required directories
    output_dir = ensure_dir('../model_coeff')
    data_dir = ensure_dir('../data')
    log_dir = ensure_dir('../logs')

    start_time = time.time()

    # Generate synthetic hydrogeological data
    nx, ny = 100, 100
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    # True lnK field (heterogeneous aquifer)
    K_true = -1 + 1.5 * np.sin(5*X) * np.cos(5*Y)

    # Observation wells
    n_obs = 25
    np.random.seed(42)
    x_obs = np.random.uniform(0, 1, n_obs)
    y_obs = np.random.uniform(0, 1, n_obs)

    # Steady-state hydraulic head field
    u_true = X + Y

    # Export MAT file for PINN inversion
    sio.savemat(os.path.join(data_dir, 'HT_synthetic.mat'), {
        'K_true': K_true,
        'u_true': u_true,
        'X': X,
        'Y': Y,
        'x_obs': x_obs,
        'y_obs': y_obs
    })

    # Export TXT files for result comparison
    np.savetxt(os.path.join(output_dir, 'K_true.txt'), K_true.ravel())
    np.savetxt(os.path.join(output_dir, 'u_true.txt'), u_true.ravel())
    np.savetxt(os.path.join(output_dir, 'X.txt'), X.ravel())
    np.savetxt(os.path.join(output_dir, 'Y.txt'), Y.ravel())
    np.savetxt(os.path.join(output_dir, 'x_obs.txt'), x_obs)
    np.savetxt(os.path.join(output_dir, 'y_obs.txt'), y_obs)

    logging.info(f"Forward simulation completed in {time.time()-start_time:.2f} seconds")
    logging.info(f"Synthetic dataset saved to: {os.path.abspath(os.path.join(data_dir, 'HT_synthetic.mat'))}")
    logging.info(f"True field data saved to: {os.path.abspath(output_dir)}")

if __name__ == '__main__':
    main()