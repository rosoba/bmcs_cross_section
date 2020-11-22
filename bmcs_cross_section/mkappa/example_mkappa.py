import matplotlib.pyplot as plt
from bmcs_cross_section.mkappa.mkappa import MKappa
from bmcs_cross_section.cs_design.cs_shape import TShapeCS, RectangleCS
from bmcs_cross_section.cs_design import CSDesign
import numpy as np


def run_example_with_t_section_and_custom_params():
    # TODO->Homam: debug MomentCurvature for t-section results

    H = 666
    # Defining a variable width (T-section as an example)
    b_w = 50
    b_f = 500
    h_w = 0.85 * H
    tshape = TShapeCS(H=H, B_w=b_w, B_f=b_f, H_w=h_w)

    mc = MKappa(n_kappa=100, n_m=100)
    mc.cross_section_shape = tshape

    # Material parameters [mm], [N/mm2]
    mc.E_ct = 24000
    mc.E_cc = 24000
    mc.eps_cr = 0.000125
    mc.eps_cy = 0.0010625  # 8.5 * eps_cr_
    mc.eps_cu = 0.0035
    mc.eps_tu = 0.02
    mc.mu = 0.33

    # 2 layers reinforcement details
    mc.A_j = np.array([250, 0])  # A_j[0] for tension steel / A_j[1] for compression steel
    mc.z_j = np.array([0.1 * mc.H, 0.9 * mc.H])
    mc.E_j = np.array([210000, 210000])
    mc.eps_sy_j = np.array([0.002, 0.002])

    # If plot_norm is used, use the following:
    # mc.kappa_range = (0, mc.kappa_cr * 100, 100

    mc.kappa_slider = -0.00001 # corresponds to idx = 25
    mc.low_kappa = -0.00002
    mc.high_kappa = 0.00002

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    mc.plot_mk_and_stress_profile(ax1, ax2)
    plt.show()


if __name__ == '__main__':
    run_example_with_t_section_and_custom_params()

    # run_example_rectangle_Yang2010_R13C_1()