import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from bmcs_cross_section.mkappa.mkappa_ import MKappa, ModelData

# Yang2010, http://dx.doi.org/10.1016/j.engstruct.2010.07.017
def get_params():
    beta_tu = 160  # SUPPOSED
    b = 180
    h = 270
    E = 46700
    E_s = 200000
    f_sy = 600
    eps_cr = 0.000384
    rho = 0.009
    v_f = 0
    omega = 10.7
    psi = 7.8
    mu = 0.55
    alpha = 0.87
    lambda_cu = 11.7
    """This method converts the params from the type of the paper to the type that can be directly used in the model"""
    return dict(
        # Substituting formulas:
        b=b,
        h=h,
        E_cc=E,
        E_ct=E,
        mu=mu,
        eps_cr=eps_cr,
        eps_cy=omega * eps_cr,
        eps_cu=lambda_cu * eps_cr,
        eps_tu=beta_tu * eps_cr,
        eps_sy_j=[f_sy / E_s], # or eps_sy_j=psi * eps_cr,
        E_j=[E_s],
        z_j=[h * (1 - alpha)],
        A_j=[rho * b * h]
    )

def run_example_rectangle_Yang2010_R13C_1():
    mc = MKappa(idx=25, n_m=100)

    mc.model_data = ModelData(**get_params())

    print(mc.model_data.eps_tu, mc.model_data.h, mc.model_data.b, mc.model_data.eps_cr)

    # mc.model_data.h = 270
    # mc.model_data.b = 180
    #
    # mc.model_data.E_ct = 46680 # supposed equal to E_cc!!
    # mc.model_data.E_cc = 46680
    # mc.model_data.eps_cr = 0.000125
    # mc.model_data.eps_cy = 0.0010625
    # mc.model_data.eps_cu = 0.0035
    # mc.model_data.eps_tu = 0.02
    # mc.model_data.mu = 0.33
    #
    # # 2 layers reinforcement details
    # mc.model_data.A_j = np.array([250, 0])  # A_j[0] for tension steel / A_j[1] for compression steel
    # mc.model_data.z_j = np.array([0.1 * model_data.h, 0.9 * model_data.h])
    # mc.model_data.E_j = np.array([210000, 210000])

    mc.kappa_range = (0, 0.0001, 100)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    mc.plot(ax1, ax2)
    plt.show()


def run_example_with_default_params():
    mc = MKappa(idx=25, n_m=100)
    mc.model_data.h = 600
    mc.model_data.b = 200
    mc.kappa_range = (-0.00002, 0.00002, 100)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    mc.plot(ax1, ax2)
    plt.show()


def run_example_with_t_section_and_custom_params():
    mc = MKappa(idx=25, n_m=100)
    model_data = ModelData()

    # Material parameters [mm], [N/mm2]
    model_data.h = 666
    model_data.E_ct = 24000
    model_data.E_cc = 24000
    model_data.eps_cr = 0.000125
    model_data.eps_cy = 0.0010625  # 8.5 * eps_cr_
    model_data.eps_cu = 0.0035
    model_data.eps_tu = 0.02
    model_data.mu = 0.33

    # 2 layers reinforcement details
    model_data.A_j = np.array([250, 0])  # A_j[0] for tension steel / A_j[1] for compression steel
    model_data.z_j = np.array([0.1 * model_data.h, 0.9 * model_data.h])
    model_data.E_j = np.array([210000, 210000])
    model_data.eps_sy_j = np.array([0.002, 0.002])

    # Defining a variable width (T-section as an example)
    b_w = 50
    b_f = 500
    h_w = 0.85 * model_data.h
    # Beam width b as a function of the height z (the sympy z symbol in MomentCurvatureSymbolic is used)
    model_data.b = sp.Piecewise((b_w, mc.mcs.z < h_w), (b_f, mc.mcs.z >= h_w))

    mc.model_data = model_data

    # If plot_norm is used, use the following:
    # mc.kappa_range = (0, mc.kappa_cr * 100, 100
    mc.kappa_range = (-0.00002, 0.00002, 100)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    mc.plot(ax1, ax2)
    plt.show()


if __name__ == '__main__':
    # run_example_with_default_params()
    run_example_with_t_section_and_custom_params()

    # run_example_rectangle_Yang2010_R13C_1()