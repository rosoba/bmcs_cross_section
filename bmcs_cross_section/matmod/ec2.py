import numpy as np


class EC2:

    # 1. Concrete strength values according to EC2 - Table 3.1
    # Units: all strength and E-modulus values for input and output are in [N/mm^2]

    @staticmethod
    def get_f_cm(f_ck):
        return f_ck + 8

    @staticmethod
    def get_f_ctm(f_ck):
        f_cm = f_ck + 8
        return np.where(f_ck <= 50, 0.3 * f_ck ** (2 / 3), 2.12 * np.log(1 + (f_cm / 10)))

    @staticmethod
    def get_f_ctk_0_05(f_ck):
        f_ctm = EC2.get_f_ctm(f_ck)
        return 0.7 * f_ctm

    @staticmethod
    def get_f_ctk_0_95(f_ck):
        f_ctm = EC2.get_f_ctm(f_ck)
        return 1.3 * f_ctm

    @staticmethod
    def get_E_cm(f_ck):
        f_cm = EC2.get_f_cm(f_ck)
        return 22000 * (f_cm / 10) ** 0.3

    @staticmethod
    def get_eps_c1(f_ck):
        f_cm = EC2.get_f_cm(f_ck)
        eps_c1 = 0.7 * f_cm ** 0.31
        eps_c1 = np.where(eps_c1 > 2.8, 2.8, eps_c1)
        return 0.001 * eps_c1

    @staticmethod
    def get_eps_cu1(f_ck):
        f_cm = EC2.get_f_cm(f_ck)
        return 0.001 * np.where(f_ck <= 50, 3.5, 2.8 + 27 * ((98 - f_cm) / 100) ** 4)

    @staticmethod
    def get_eps_c2(f_ck):
        return 0.001 * np.where(f_ck <= 50, 2, 2 + 0.085 * (f_ck - 50) ** 0.53)

    @staticmethod
    def get_eps_cu2(f_ck):
        return 0.001 * np.where(f_ck <= 50, 3.5, 2.6 + 35 * ((90 - f_ck) / 100) ** 4)

    @staticmethod
    def get_eps_c3(f_ck):
        return 0.001 * np.where(f_ck <= 50, 1.75, 1.75 + 0.55 * ((f_ck - 50) / 40))

    @staticmethod
    def get_eps_cu3(f_ck):
        return 0.001 * np.where(f_ck <= 50, 3.5, 2.6 + 35 * ((90 - f_ck) / 100) ** 4)
