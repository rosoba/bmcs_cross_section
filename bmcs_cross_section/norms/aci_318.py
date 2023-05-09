import numpy as np

from bmcs_cross_section.norms.ec2 import EC2


class ACI318:
    """
    The American steel reinforced concrete code
    """
    @staticmethod
    def get_beta_1(f_c):
        return 0.85 if f_c <= 28 else max((0.85 - 0.05 * (f_c - 28) / 7), 0.65)

    @staticmethod
    def get_M_n(A_s=50, f_y=500, f_c=48, b=200, d=280):
        a = A_s * f_y / (0.85 * f_c * b)
        return A_s * f_y * (d - a / 2)