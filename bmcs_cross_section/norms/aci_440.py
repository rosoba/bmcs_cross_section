import numpy as np

from bmcs_cross_section.norms.ec2 import EC2


class ACI440:
    """
    The American FRP reinforced concrete code
    """
    @staticmethod
    def get_beta_1(f_c):
        return 0.85 if f_c <= 28 else max((0.85 - 0.05 * (f_c - 28) / 7), 0.65)

    @staticmethod
    def get_rho_balanced(f_c, f_fu, E_f):
        """
        :param f_c [MPa]: compressive strength of concrete (typically specified compressive strength f'c = f_ck)
        :param f_fu [MPa]: tensile strength of FRP reinf.
        :param E_f [MPa]: e-modulus of FRP reinf.
        :return: Balanced FRP reinforcement ratio
        """
        eps_cu = ACI440.get_eps_cu()
        beta_1 = ACI440.get_beta_1(f_c)
        rho_fb = 0.85 * beta_1 * (f_c / f_fu) * (E_f * eps_cu / (E_f * eps_cu + f_fu))
        return rho_fb


    @staticmethod
    def get_M_n(A_f=50, f_fu=2500, E_f=158000, f_c=48, b=200, d=280):
        """
        :param f_c [MPa]: compressive strength of concrete (typically specified compressive strength f'c = f_ck)
        :param f_fu [MPa]: tensile strength of FRP reinf.
        :param E_f [MPa]: e-modulus of FRP reinf.
        :return: Balanced FRP reinforcement ratio
        """
        rho = A_f / (b * d)
        rho_balanced = ACI440.get_rho_balanced(f_c, f_fu, E_f)
        eps_cu = ACI440.get_eps_cu()
        beta_1 = ACI440.get_beta_1(f_c)
        if rho <= rho_balanced:
            eps_fu = f_fu / E_f
            c_b = (eps_cu / (eps_cu + eps_fu)) * d
            M_n = A_f * f_fu * (d - beta_1 * c_b / 2) / 1e6
        else:
            f_f = np.minimum(
                np.sqrt(((E_f * eps_cu) ** 2) / 4 + 0.85 * beta_1 * f_c * E_f * eps_cu / rho) - 0.5 * E_f * eps_cu,
                f_fu)
            a = A_f * f_f / (0.85 * f_c * b)
            M_n = A_f * f_f * (d - a / 2) / 1e6
        return M_n


    @staticmethod
    def get_psi_f(rho=0.005, f_fu=2500, E_f=158000, f_c=48):
        rho_balanced = ACI440.get_rho_balanced(f_c, f_fu, E_f)
        eps_cu = ACI440.get_eps_cu()
        beta_1 = ACI440.get_beta_1(f_c)
        if rho <= rho_balanced:
            return 1
        else:
            f_f = np.minimum(
                np.sqrt(((E_f * eps_cu) ** 2) / 4 + 0.85 * beta_1 * f_c * E_f * eps_cu / rho) - 0.5 * E_f * eps_cu,
                f_fu)
            return f_f / f_fu


    @staticmethod
    def get_psi_c(rho=0.005, A_f=50, f_fu=2500, E_f=158000, f_c=48, b=200, d=280):
        rho_balanced = ACI440.get_rho_balanced(f_c, f_fu, E_f)
        eps_cu = ACI440.get_eps_cu()
        beta_1 = ACI440.get_beta_1(f_c)
        if rho <= rho_balanced:
            eps_fu = f_fu / E_f
            c_b = (eps_cu / (eps_cu + eps_fu)) * d
        else:
            return 1

    @staticmethod
    def get_eps_cu():
        return 0.003
