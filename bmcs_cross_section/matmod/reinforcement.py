from .matmod import MatMod
import bmcs_utils.api as bu
import sympy as sp
import numpy as np
import traits.api as tr

class ReinfMatMod(MatMod):
    pass

class SteelReinfMatModSymbExpr(bu.SymbExpr):
    """Piecewise linear concrete material law
    """
    eps = sp.Symbol('eps', real=True)

    eps_sy, E_s = sp.symbols('varepsilon_sy, E_s', real=True, nonnegative=True)

    # steel_material_factor = 1. / 1.15
    steel_material_factor = 1

    sig = steel_material_factor * sp.Piecewise(
        (-E_s * eps_sy, eps < -eps_sy),
        (E_s * eps, eps < eps_sy),
        (E_s * eps_sy, eps >= eps_sy)
    )

    symb_model_params = ('E_s', 'eps_sy')

    symb_expressions = [
        ('sig', ('eps',)),
    ]


class SteelReinfMatMod(ReinfMatMod, bu.InjectSymbExpr):
    name = 'Steel'

    symb_class = SteelReinfMatModSymbExpr

    E_s = bu.Float(200000, MAT=True, desc='E modulus of steel')
    f_sy = bu.Float(500, MAT=True, desc='steel yield stress')

    eps_sy = tr.Property(bu.Float, depends_on='+MAT')
    @tr.cached_property
    def _get_eps_sy(self):
        return self.f_sy / self.E_s

    ipw_view = bu.View(
        bu.Item('E_s', latex=r'E_\mathrm{s} \mathrm{[N/mm^{2}]}'),
        bu.Item('f_sy', latex=r'f_\mathrm{sy} \mathrm{[N/mm^{2}]}'),
        bu.Item('eps_sy', latex=r'\varepsilon_\mathrm{sy} \mathrm{[-]}'),
    )

    def get_eps_plot_range(self):
        return np.linspace(- 1.1*self.eps_sy, 1.1*self.eps_sy,300)

    def get_sig(self,eps):
        return self.symb.get_sig(eps)

class CarbonReinfMatModSymbExpr(bu.SymbExpr):
    """Piecewise linear concrete material law
    """
    eps = sp.Symbol('eps', real=True)

    f_t, E = sp.symbols('f_t, E', real=True, nonnegative=True)

    # carbon_material_factor = 1. / 1.5
    carbon_material_factor = 1

    sig = carbon_material_factor * sp.Piecewise(
        (0, eps < 0),
        (E * eps, eps < f_t/E),
        (f_t - E * (eps - f_t/E), eps < 2 * f_t/E),
        (0, True)
    )

    symb_model_params = ('E', 'f_t')

    symb_expressions = [
        ('sig', ('eps',)),
    ]


class CarbonReinfMatMod(ReinfMatMod, bu.InjectSymbExpr):

    name = 'Steel'

    symb_class = CarbonReinfMatModSymbExpr

    E = bu.Float(200000, MAT=True, desc='E modulus of carbon')
    f_t = bu.Float(2000, MAT=True, desc='carbon breaking stress')

    eps_cr = tr.Property(bu.Float, depends_on='+MAT')
    @tr.cached_property
    def _get_eps_cr(self):
        return self.f_t / self.E

    ipw_view = bu.View(
        bu.Item('E', latex=r'E \mathrm{[N/mm^{2}]}'),
        bu.Item('f_t', latex=r'f_\mathrm{t} \mathrm{[N/mm^{2}]}'),
        bu.Item('eps_cr', latex=r'\varepsilon_\mathrm{cr} \mathrm{[-]}'),
    )

    def get_eps_plot_range(self):
        return np.linspace(- 0.1*self.eps_cr, 1.1*self.eps_cr,300)

    def get_sig(self,eps):
        return self.symb.get_sig(eps)
