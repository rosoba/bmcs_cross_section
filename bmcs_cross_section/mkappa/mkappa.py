import numpy as np
import sympy as sp
import traits.api as tr
from bmcs_cross_section.cs_design import CrossSectionDesign
from scipy.optimize import root
from bmcs_utils.api import \
    InteractiveModel, Item, View, mpl_align_xaxis, \
    SymbExpr, InjectSymbExpr, Float, Int, FloatRangeEditor

import enum


class ReinforcementType(enum.Enum):
    STEEL, CARBON = range(2)


class MKappaSymbolic(SymbExpr):
    """This class handles all the symbolic calculations
    so that the class MomentCurvature doesn't use sympy ever
    """
    # -------------------------------------------------------------------------
    # Symbolic derivation of expressions
    # -------------------------------------------------------------------------
    # kappa = sp.Symbol('kappa', real=True, nonpositive=True)
    # eps_top = sp.symbols('varepsilon_top', real=True) # , nonpositive=True)
    # eps_bot = sp.symbols('varepsilon_bot', real=True, nonnegative =True)
    kappa = sp.Symbol('kappa', real=True)
    eps_top = sp.symbols('varepsilon_top', real=True)
    eps_bot = sp.symbols('varepsilon_bot', real=True)
    b, h, z = sp.symbols('b, h, z', real=True, nonnegative=True)
    eps_sy, E_s = sp.symbols('varepsilon_sy, E_s', real=True, nonnegative=True)
    eps = sp.Symbol('varepsilon', real=True)

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

    # -------------------------------------------------------------------------
    # Symbolic derivation of expressions
    # -------------------------------------------------------------------------
    # Linear profile of strain over the cross section height
    eps_z_ = eps_bot + z * (eps_top - eps_bot) / h
    eps_top_solved = {eps_top: sp.solve(kappa + eps_z_.diff(z), eps_top)[0]}
    eps_z = eps_z_.subs(eps_top_solved)

    sig_c_eps = sp.Piecewise(
        (0, eps < eps_cu),
        (E_cc * eps_cy, eps < eps_cy),
        (E_cc * eps, eps < 0),
        (E_ct * eps, eps < eps_cr),
        (mu * E_ct * eps_cr, eps < eps_tu),
        (0, True) #  eps >= eps_tu)
    )

    # Stress over the cross section height
    # sig_c_z_ = sig_c_eps.subs(eps, eps_z)
    sig_c_z_ = sig_c_eps.subs(eps, eps_z_) # this was like this originally

    # Substitute eps_top to get sig as a function of (kappa, eps_bot, z)
    sig_c_z = sig_c_z_.subs(eps_top_solved)

    # Reinforcement constitutive law
    sig_s_eps = tr.Property()

    def _get_sig_s_eps(self):
        if self.model.reinforcement_type == ReinforcementType.STEEL:
            sig_s_eps = sp.Piecewise(
                (-self.E_s * self.eps_sy, self.eps < -self.eps_sy),
                (self.E_s * self.eps, self.eps < self.eps_sy),
                (self.E_s * self.eps_sy, self.eps >= self.eps_sy)
            )
        elif self.model.reinforcement_type == ReinforcementType.CARBON:
            sig_s_eps = sp.Piecewise(
                (0, self.eps < 0),
                (self.E_s * self.eps, self.eps < self.eps_sy),
                (self.E_s * self.eps_sy - self.E_s * (self.eps - self.eps_sy), self.eps < 2 * self.eps_sy),
                (0, True)
            )
        else:
            raise NameError('There\'s no reinforcement type with the name ' + self.model.reinforcement_typ)
        return sig_s_eps

    # ----------------------------------------------------------------
    # SymbExpr protocol: Parameter names to be fetched from the model
    # ----------------------------------------------------------------
    symb_model_params = ('E_ct', 'E_cc', 'eps_cr', 'eps_cy', 'eps_cu', 'mu', 'eps_tu')

    symb_expressions = [
        ('eps_z', ('kappa', 'eps_bot', 'z')),
        ('sig_c_eps', ('eps',)),
#        ('sig_c_z', ('kappa', 'eps_bot', 'z')),
        ('sig_s_eps', ('eps', 'E_s', 'eps_sy')),
    ]


class MKappa(InteractiveModel, InjectSymbExpr):
    """Class returning the moment curvature relationship."""
    name = 'Moment-Curvature'

    reinforcement_type = ReinforcementType.STEEL

    ipw_view = View(
        Item('low_kappa', latex=r'\text{Low}~\kappa'),
        Item('high_kappa', latex=r'\text{High}~\kappa'),
        Item('n_kappa', latex='n_{\kappa}'),
        Item('n_m', latex='n_m \mathrm{[mm]}'),
        Item('kappa_slider', latex='\kappa',
             editor=FloatRangeEditor(low_name='low_kappa',
                                     high_name='high_kappa',
                                     n_steps_name='n_kappa')
             )
    )

    symb_class = MKappaSymbolic
    cs_design = tr.Instance(CrossSectionDesign, ())

    # Use PrototypedFrom only when the prototyped object is a class (The prototyped attribute behaves similarly
    # to a delegated attribute, until it is explicitly changed; from that point forward, the prototyped attribute
    # changes independently from its prototype.) (it's kind of like tr.DelegatesTo('cs_design.cross_section_shape'))
    cross_section_shape = tr.PrototypedFrom('cs_design', 'cross_section_shape')
    cross_section_layout = tr.PrototypedFrom('cs_design', 'cross_section_layout')
    matrix = tr.PrototypedFrom('cross_section_layout', 'matrix')
    reinforcement = tr.PrototypedFrom('cross_section_layout', 'reinforcement')

    # Geometry
    H = tr.DelegatesTo('cross_section_shape')

    # Concrete
    E_ct = tr.DelegatesTo('matrix')
    E_cc = tr.Float(20000, MAT=True) # tr.DelegatesTo('matrix')
    eps_cr = tr.DelegatesTo('matrix')
    eps_tu = tr.DelegatesTo('matrix')
    mu = tr.DelegatesTo('matrix')

    eps_cy = tr.DelegatesTo('matrix')
    eps_cu = tr.DelegatesTo('matrix')

    # Reinforcement
    z_j = tr.DelegatesTo('reinforcement')
    A_j = tr.DelegatesTo('reinforcement')
    E_j = tr.DelegatesTo('reinforcement')
    eps_sy_j = tr.DelegatesTo('reinforcement')

    DEPSTR = '+BC, +MAT, cross_section_shape.+GEO, matrix.+MAT, reinforcement.+MAT'

    n_m = Int(100, DSC=True)

    # @todo: fix the dependency - `H` should be replaced by _GEO
    z_m = tr.Property(depends_on='n_m, H')

    @tr.cached_property
    def _get_z_m(self):
        return np.linspace(0, self.H, self.n_m)

    low_kappa = Float(-0.0001, BC=True)
    high_kappa = Float(0.0001, BC=True)
    n_kappa = Int(100, BC=True)

    kappa_slider = Float(0)

    idx = tr.Property(depends_on='kappa_slider')

    @tr.cached_property
    def _get_idx(self):
        ks = self.kappa_slider
        idx = np.argmax(ks <= self.kappa_t)
        return idx

    kappa_t = tr.Property(tr.Array(np.float_), depends_on='+BC')
    '''Curvature values for which the bending moment must be found
    '''

    @tr.cached_property
    def _get_kappa_t(self):
        return np.linspace(self.low_kappa, self.high_kappa, self.n_kappa)

    # Normal force in steel (tension and compression)
    def get_N_s_tj(self, kappa_t, eps_bot_t):
        eps_z_tj = self.symb.get_eps_z(
            kappa_t[:, np.newaxis], eps_bot_t[:, np.newaxis],
            self.z_j[np.newaxis, :]
        )
        sig_s_tj = self.symb.get_sig_s_eps(
            eps_z_tj, self.E_j, self.eps_sy_j
        )
        N_s_tj = np.einsum('j,tj->tj', self.A_j, sig_s_tj)
        return N_s_tj

    def get_sig_c_z(self, kappa_t, eps_bot_t, z_tm):
        eps_z = self.symb.get_eps_z(kappa_t[:, np.newaxis], eps_bot_t[:, np.newaxis], z_tm)
        sig_c_z = self.symb.get_sig_c_eps(eps_z)
        return sig_c_z

    # Normal force in concrete (tension and compression)
    def get_N_c_t(self, kappa_t, eps_bot_t):
        z_tm = self.z_m[np.newaxis, :]
        b_z_m = self.cross_section_shape.get_b(z_tm)
        #N_z_tm1 = b_z_m * self.symb.get_sig_c_z(kappa_t[:, np.newaxis], eps_bot_t[:, np.newaxis], z_tm)
        N_z_tm2 = b_z_m * self.get_sig_c_z(kappa_t, eps_bot_t, z_tm)
        return np.trapz(N_z_tm2, x=z_tm, axis=-1)

    def get_N_t(self, kappa_t, eps_bot_t):
        N_s_t = np.sum(self.get_N_s_tj(kappa_t, eps_bot_t), axis=-1)
        N_c_t = self.get_N_c_t(kappa_t, eps_bot_t)
        return N_c_t + N_s_t

    # SOLVER: Get eps_bot to render zero force

    eps_bot_t = tr.Property(depends_on=DEPSTR)
    r'''Resolve the tensile strain to get zero normal force for the prescribed curvature'''

    @tr.cached_property
    def _get_eps_bot_t(self):
        res = root(lambda eps_bot_t: self.get_N_t(self.kappa_t, eps_bot_t),
                   0.0000001 + np.zeros_like(self.kappa_t), tol=1e-6)
        if not res.success:
            print('No solution', res.message)
        return res.x

    # POSTPROCESSING

    kappa_cr = tr.Property(depends_on=DEPSTR)
    '''Curvature at which a critical strain is attained at the eps_bot'''

    @tr.cached_property
    def _get_kappa_cr(self):
        res = root(lambda kappa: self.get_N_t(kappa, self.eps_cr),
                   0.0000001 + np.zeros_like(self.eps_cr), tol=1e-10)
        if not res.success:
            print('No solution', res.message)
        return res.x

    M_s_t = tr.Property(depends_on=DEPSTR)
    '''Bending moment (steel)
    '''

    @tr.cached_property
    def _get_M_s_t(self):
        eps_z_tj = self.symb.get_eps_z(
            self.kappa_t[:, np.newaxis], self.eps_bot_t[:, np.newaxis],
            self.z_j[np.newaxis, :]
        )
        sig_z_tj = self.symb.get_sig_s_eps(
            eps_z_tj, self.E_j, self.eps_sy_j
        )
        return -np.einsum('j,tj,j->t', self.A_j, sig_z_tj, self.z_j)

    M_c_t = tr.Property(depends_on=DEPSTR)
    '''Bending moment (concrete)
    '''

    @tr.cached_property
    def _get_M_c_t(self):
        z_tm = self.z_m[np.newaxis, :]
        b_z_m = self.cross_section_shape.get_b(z_tm)
        #N_z_tm1 = b_z_m * self.symb.get_sig_c_z(self.kappa_t[:, np.newaxis], self.eps_bot_t[:, np.newaxis], z_tm)
        N_z_tm2 = b_z_m * self.get_sig_c_z(self.kappa_t, self.eps_bot_t, z_tm)
        return -np.trapz(N_z_tm2 * z_tm, x=z_tm, axis=-1)

    M_t = tr.Property(depends_on=DEPSTR)
    '''Bending moment
    '''

    @tr.cached_property
    def _get_M_t(self):
        return self.M_c_t + self.M_s_t

    N_s_tj = tr.Property(depends_on=DEPSTR)
    '''Normal forces (steel)
    '''

    @tr.cached_property
    def _get_N_s_tj(self):
        return self.get_N_s_tj(self.kappa_t, self.eps_bot_t)

    eps_tm = tr.Property(depends_on=DEPSTR)
    '''strain profiles
    '''

    @tr.cached_property
    def _get_eps_tm(self):
        return self.get_eps_z(self.kappa_t[:, np.newaxis],
                              self.eps_bot_t[:, np.newaxis], self.z_m[np.newaxis, :])

    sig_tm = tr.Property(depends_on=DEPSTR)
    '''strain profiles
    '''

    @tr.cached_property
    def _get_sig_tm(self):
        return self.get_sig_c_z(
            self.kappa_t, self.eps_bot_t, self.z_m[np.newaxis, :]
        )

    M_norm = tr.Property(depends_on=DEPSTR)
    '''
    '''

    @tr.cached_property
    def _get_M_norm(self):
        # Section modulus @TODO optimize W for var b
        W = (self.b * self.H ** 2) / 6
        sig_cr = self.E_ct * self.eps_cr
        return W * sig_cr

    kappa_norm = tr.Property()

    def _get_kappa_norm(self):
        return self.kappa_cr

    inv_M_kappa = tr.Property(depends_on=DEPSTR)
    '''Return the inverted data points
    '''
    @tr.cached_property
    def _get_inv_M_kappa(self):
        """cut off the descending tails"""
        M_t = self.M_t
        I_max = np.argmax(M_t)
        I_min = np.argmin(M_t)
        M_I = self.M_t[I_min:I_max + 1]
        kappa_I = self.kappa_t[I_min:I_max + 1]
        # find the index corresponding to zero kappa
        idx = np.argmax(0 <= self.kappa_t)
        # and modify the values such that the
        # Values of moment are non-descending
        M_plus = M_I[idx:]
        M_diff = M_plus[:, np.newaxis] - M_plus[np.newaxis, :]
        n_ij = len(M_plus)
        ij = np.mgrid[0:n_ij:1, 0:n_ij:1]
        M_diff[np.where(ij[1] >= ij[0])] = 0
        i_x = np.argmin(M_diff, axis=1)
        M_I[idx:] = M_plus[i_x]
        return M_I, kappa_I

    def get_kappa_M(self, M):
        M_I, kappa_I = self.inv_M_kappa
        return np.interp(M, M_I, kappa_I)

    def plot_norm(self, ax1, ax2):
        idx = self.idx
        ax1.plot(self.kappa_t / self.kappa_norm, self.M_t / self.M_norm)
        ax1.plot(self.kappa_t[idx] / self.kappa_norm, self.M_t[idx] / self.M_norm, marker='o')
        ax2.barh(self.z_j, self.N_s_tj[idx, :], height=2, color='red', align='center')
        # ax2.fill_between(eps_z_arr[idx,:], z_arr, 0, alpha=0.1);
        ax3 = ax2.twiny()
        #        ax3.plot(self.eps_tm[idx, :], self.z_m, color='k', linewidth=0.8)
        ax3.plot(self.sig_tm[idx, :], self.z_m)
        ax3.axvline(0, linewidth=0.8, color='k')
        ax3.fill_betweenx(self.z_m, self.sig_tm[idx, :], 0, alpha=0.1)
        mpl_align_xaxis(ax2, ax3)

    M_scale = Float(1e+6)

    def plot(self, ax1, ax2, ax3):
        self.plot_mk_and_stress_profile(ax1, ax2)

        M, kappa = self.inv_M_kappa
        ax3.plot(M / self.M_scale, kappa)
        ax3.set_xlabel('Moment [kNm]')
        ax3.set_ylabel('Curvature[mm$^{-1}$]')

    @staticmethod
    def subplots(fig):
        ax1, ax2, ax3 = fig.subplots(1, 3)
        return ax1, ax2, ax3

    def update_plot(self, axes):
        self.plot(*axes)

    def plot_mk_and_stress_profile(self, ax1, ax2):
        self.plot_mk(ax1)

        idx = self.idx
        ax1.plot(self.kappa_t[idx], self.M_t[idx] / self.M_scale, color='orange', marker='o')
        ax2.barh(self.z_j, self.N_s_tj[idx, :], height=6, color='red', align='center')

        ax22 = ax2.twiny()
        ax22.plot(self.sig_tm[idx, :], self.z_m)
        ax22.axvline(0, linewidth=0.8, color='k')
        ax22.fill_betweenx(self.z_m, self.sig_tm[idx, :], 0, alpha=0.1)
        mpl_align_xaxis(ax2, ax22)

    def plot_mk(self, ax1):
        ax1.plot(self.kappa_t, self.M_t / self.M_scale, label='bmcs_cs_mkappa')
        ax1.set_ylabel('Moment [kNm]')
        ax1.set_xlabel('Curvature [mm$^{-1}$]')
        ax1.legend()
