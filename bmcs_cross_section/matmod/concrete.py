import bmcs_utils.api as bu
import sympy as sp
import numpy as np
import traits.api as tr

from .matmod import MatMod

class ConcreteMatMod(MatMod):
    pass


class PWLConcreteMatModSymbExpr(bu.SymbExpr):
    """Piecewise linear concrete material law
    """
    eps = sp.Symbol('eps', real=True)
    # -------------------------------------------------------------------------
    # Model parameters
    # -------------------------------------------------------------------------
    E_ct, E_cc, eps_cr, eps_tu, mu = sp.symbols(
        r'E_ct, E_cc, varepsilon_cr, varepsilon_tu, mu', real=True,
        nonnegative=True
    )
    eps_cy, eps_cu = sp.symbols(
        r'varepsilon_cy, varepsilon_cu',
        real=True, nonpositive=True
    )

    # TODO - declare it to a material parameter
    # concrete_material_factor = 0.85 / 1.5
    concrete_material_factor = 1

    sig = concrete_material_factor * sp.Piecewise(
        (0, eps < eps_cu),
        (E_cc * eps_cy, eps < eps_cy),
        (E_cc * eps, eps < 0),
        (E_ct * eps, eps < eps_cr),
        (mu * E_ct * eps_cr, eps < eps_tu),
        (0, True)
    )

    symb_model_params = ('E_ct', 'E_cc', 'eps_cr', 'eps_cy', 'eps_cu',
                         'mu', 'eps_tu')

    symb_expressions = [
        ('sig', ('eps',)),
    ]


class PWLConcreteMatMod(ConcreteMatMod, bu.InjectSymbExpr):
    name = 'Concrete PWL'

    symb_class = PWLConcreteMatModSymbExpr

    E_ct = bu.Float(24000, MAT=True, desc='E modulus of matrix on tension')
    E_cc = bu.Float(25000, MAT=True, desc='E modulus of matrix on compression')
    eps_cr = bu.Float(0.001, MAT=True, desc='Matrix cracking strain')
    _eps_cy = bu.Float(-0.003, MAT=True)
    _eps_cu = bu.Float(-0.01, MAT=True)

    # Enforcing negative values for eps_cu and eps_cy
    eps_cy = tr.Property(desc='Matrix compressive yield strain')
    def _set_eps_cy(self, value):
        self._eps_cy = value
    def _get_eps_cy(self):
        return -np.fabs(self._eps_cy)

    eps_cu = tr.Property(desc='Ultimate matrix compressive strain')
    def _set_eps_cu(self, value):
        self._eps_cu = value
    def _get_eps_cu(self):
        return -np.fabs(self._eps_cu)

    eps_tu = bu.Float(0.003, MAT=True, desc='Ultimate matrix tensile strain')

    mu = bu.Float(0.33, MAT=True, desc='Post crack tensile strength ratio (represents how much strength is left after \
                                    the crack because of short steel fibers in the mixture)')

    ipw_view = bu.View(
        bu.Item('E_ct', latex=r'E_\mathrm{ct} \mathrm{[N/mm^{2}]}'),
        bu.Item('E_cc', latex=r'E_\mathrm{cc} \mathrm{[N/mm^{2}]}'),
        bu.Item('eps_cr', latex=r'\varepsilon_{cr}'),
        bu.Item('eps_cy', latex=r'\varepsilon_{cy}', editor=bu.FloatEditor()),
        bu.Item('eps_cu', latex=r'\varepsilon_{cu}', editor=bu.FloatEditor()),
        bu.Item('eps_tu', latex=r'\varepsilon_{tu}'),
        bu.Item('mu', latex=r'\mu')
    )

    def get_eps_plot_range(self):
        return np.linspace(1.1*self.eps_cu, 1.1*self.eps_tu,300)

    def get_sig(self,eps):
        return self.symb.get_sig(eps)


class EC2ConcreteMatModSymbExpr(bu.SymbExpr):
    """Piecewise linear concrete material law
    """
    eps = sp.Symbol('eps', real=True)

    # -------------------------------------------------------------------------
    # Model parameters
    # -------------------------------------------------------------------------
    E_ct, E_cc, eps_cr, eps_tu, mu = sp.symbols(
        r'E_ct, E_cc, varepsilon_cr, varepsilon_tu, mu', real=True,
        nonnegative=True
    )
    eps_cy, eps_cu = sp.symbols(
        r'varepsilon_cy, varepsilon_cu',
        real=True, nonpositive=True
    )
    # -------------- Concrete material law according to EC2 Eq. (3.14) ------------------
    f_cm = sp.Symbol('f_cm', real=True)
    f_cd = sp.Symbol('f_cd', real=True)
    n = sp.Symbol('n', real=True)

    k = 1.05 * E_cc * sp.Abs(eps_cy) / f_cm
    eta = eps / eps_cy
    sig_c = f_cm * (k * eta - eta ** 2) / (1 + eta * (k - 2))

    # with continuous curve until sig_c = 0
    # eps_at_sig_0 = sp.solve(sig_c, self.eps)[1]
    # sig_c_eps = -sp.Piecewise(
    #     (0, self.eps < eps_at_sig_0),
    #     (sig_c, self.eps <= 0),  # use eps <= 0),
    #
    #     # Tension branch
    #     (-self.E_ct * self.eps, self.eps < self.eps_cr),
    #
    #     (0, True)
    # )

    # with extra line
    # eps_extra = 0.005
    # sig_c_cu = sig_c.subs(self.eps, self.eps_cu)
    # extra_line = sig_c_cu + sig_c_cu * (self.eps - self.eps_cu) / eps_extra
    # eps_at_sig_0 = sp.solve(extra_line, self.eps)[0]
    # sig_c_eps = -sp.Piecewise(
    #     (0, self.eps < eps_at_sig_0),
    #     (extra_line, self.eps < self.eps_cu),
    #     (sig_c, self.eps <= 0),
    #     (0, True)
    # )

    # with direct drop exactly like EC2 drawing
    # sig_c_eps = - sp.Piecewise(
    #     (0, self.eps < self.eps_cu),
    #     (sig_c, self.eps <= 0),
    #     (0, True)
    # )

    # EC2 eq. (3.17-3.18)
    sig_c = f_cd * (1 - (1 - eps / eps_cy) ** n)
    sig = -sp.Piecewise(
        (0, eps < eps_cu),
        (f_cd, eps < eps_cy),
        (sig_c, eps < 0),
        # Tension branch
        (-E_ct * eps, eps < eps_cr),
        # Tension branch, adding post-peak branch
        (-mu * E_ct * eps_cr, eps < eps_tu),
        (0, True)
    )

    # if self.model.apply_material_safety_factors:
    #     # sig_c_eps = sig_c_eps - 8 # F_ck = F_cm - 8
    #     sig_c_eps = self.concrete_material_factor * sig_c_eps

    symb_model_params = ('E_ct', 'E_cc', 'eps_cr', 'eps_cy', 'eps_cu',
                         'mu', 'eps_tu', 'f_cd', 'f_cm', 'n')

    symb_expressions = [
        ('sig', ('eps',)),
    ]


class EC2ConcreteMatMod(ConcreteMatMod, bu.InjectSymbExpr):
    name = 'EC2 Concrete'

    symb_class = EC2ConcreteMatModSymbExpr

    E_ct = bu.Float(24000, MAT=True, desc='E modulus of matrix on tension')
    E_cc = bu.Float(25000, MAT=True, desc='E modulus of matrix on compression')
    eps_cr = bu.Float(0.001, MAT=True, desc='Matrix cracking strain')
    _eps_cy = bu.Float(-0.003, MAT=True)
    _eps_cu = bu.Float(-0.01, MAT=True)

    # Enforcing negative values for eps_cu and eps_cy
    f_cm = bu.Float(28)

    f_cd = bu.Float(28 * 0.85 / 1.5)
    n = bu.Float(2)

    eps_cy = tr.Property(desc='Matrix compressive yield strain')
    def _set_eps_cy(self, value):
        self._eps_cy = value

    def _get_eps_cy(self):
        return -np.fabs(self._eps_cy)

    eps_cu = tr.Property(desc='Ultimate matrix compressive strain')

    def _set_eps_cu(self, value):
        self._eps_cu = value

    def _get_eps_cu(self):
        return -np.fabs(self._eps_cu)

    eps_tu = bu.Float(0.003, MAT=True, desc='Ultimate matrix tensile strain')

    mu = bu.Float(0.33, MAT=True, desc='Post crack tensile strength ratio (represents how much strength is left after \
                                    the crack because of short steel fibers in the mixture)')

    ipw_view = bu.View(
        bu.Item('E_ct', latex=r'E_\mathrm{ct} \mathrm{[N/mm^{2}]}'),
        bu.Item('E_cc', latex=r'E_\mathrm{cc} \mathrm{[N/mm^{2}]}'),
        bu.Item('eps_cr', latex=r'\varepsilon_{cr}'),
        bu.Item('eps_cy', latex=r'\varepsilon_{cy}', editor=bu.FloatEditor()),
        bu.Item('eps_cu', latex=r'\varepsilon_{cu}', editor=bu.FloatEditor()),
        bu.Item('eps_tu', latex=r'\varepsilon_{tu}'),
        bu.Item('mu', latex=r'\mu'),
        bu.Item('f_cd', latex=r'f_\mathrm{cd}'),
        bu.Item('f_cm', latex=r'f_\mathrm{cm}'),
        bu.Item('n', latex=r'\mu'),
    )

    def get_eps_plot_range(self):
        return np.linspace(self.eps_cu, self.eps_tu, 300)

    def get_sig(self, eps):
        return self.symb.get_sig(eps)

    # TODO - [HS] check if the lines below can be deleted
    #
    # Moved already to the if statement but was left here in case the if would be changed
    # sig_c_eps = concrete_material_factor * sp.Piecewise(
    #     (0, eps < eps_cu),
    #     (E_cc * eps_cy, eps < eps_cy),
    #     (E_cc * eps, eps < 0),
    #     (E_ct * eps, eps < eps_cr),
    #     (mu * E_ct * eps_cr, eps < eps_tu),
    #     (0, True) #  eps >= eps_tu)
    # )

    # Alternative with descending branch instead of sudden drop for tension
    # sig_c_eps = sp.Piecewise(
    #     (0, eps < eps_cu),
    #     (E_cc * eps_cy, eps < eps_cy),
    #     (E_cc * eps, eps < 0),
    #     (E_ct * eps, eps < eps_cr),
    #     (mu * E_ct * eps_cr * (eps - eps_tu) / (eps_cr - eps_tu), eps < eps_tu),
    #     (0, True) #  eps >= eps_tu)
    # )

    # # The following was replaced with a Property because otherwise it's not accepting sig_c_eps value
    # # Stress over the cross section height
    # # sig_c_z_ = sig_c_eps.subs(eps, eps_z)
    # sig_c_z_ = sig_c_eps.subs(eps, eps_z_) # this was like this originally
    #
    # # The following was replaced with a Property because otherwise it's not accepting sig_c_eps value
    # # Substitute eps_top to get sig as a function of (kappa, eps_bot, z)
    # sig_c_z = sig_c_z_.subs(eps_top_solved)

