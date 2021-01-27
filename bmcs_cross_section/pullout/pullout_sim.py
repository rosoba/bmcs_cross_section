'''
Created on 12.01.2016
@author: ABaktheer, RChudoba

@todo: enable recalculation after the initial offline run
@todo: reset viz adapters upon recalculation to forget their axes lims
@todo: introduce a switch for left and right supports
'''
import copy
import bmcs_utils.api as bu
from ibvpy.tfunction import LoadingScenario
from ibvpy.bcond import BCDof
from ibvpy.fets.fets1D5 import FETS1D52ULRH
from ibvpy.tmodel import IMATSEval
from ibvpy.tmodel.mats1D5.vmats1D5_bondslip1D import \
    MATSBondSlipMultiLinear, MATSBondSlipDP, \
    MATSBondSlipD, MATSBondSlipEP, MATSBondSlipFatigue
from ibvpy.api import \
    TStepBC, Hist, XDomainFEInterface1D
from ibvpy.view.reporter import RInputRecord
from scipy import interpolate as ip
from scipy.integrate import cumtrapz
from traits.api import \
    Property, Instance, cached_property, \
    HasStrictTraits, Bool, List, Float, Trait, Int, Enum, \
    Array, Button, on_trait_change, Tuple
from traitsui.api import \
    View, Item, Group
from traitsui.ui_editors.array_view_editor import ArrayViewEditor
from ibvpy.view.plot2d import Vis2D
from ibvpy.view.ui import BMCSLeafNode, itags_str, BMCSRootNode
from ibvpy.view.window import BMCSWindow, PlotPerspective

import numpy as np
import traits.api as tr



class PulloutHist(Hist, bu.InteractiveModel, Vis2D):
    name = 'History'

    record_traits = tr.List(
        ['P', 'w_0', 'w_L', ]
    )

    record_t = tr.Dict

    Pw = Tuple()

    def _Pw_default(self):
        return ([0], [0], [0])

    sig_t = List([])
    eps_t = List([])

    def init_state(self):
        super(PulloutHist, self).init_state()
        for rt in self.record_traits:
            self.record_t[rt] = [0]

    def record_timestep(self, t, U, F,
                        state_vars=None):
        t_n1 = self.tstep_source.t_n1
        for rt in self.record_traits:
            self.record_t[rt].append(getattr(self.tstep_source, rt))
        super(PulloutHist, self).record_timestep(t, U, F, state_vars)

    def get_Pw_t(self):
        c_dof = self.tstep_source.controlled_dof
        f_dof = self.tstep_source.free_end_dof
        n_t = self.n_t
        U_ti = self.U_t
        F_ti = self.F_t
        P = F_ti[:n_t, c_dof]
        w_L = U_ti[:n_t, c_dof]
        w_0 = U_ti[:n_t, f_dof]
        return P, w_0, w_L

    def get_u_p(self):
        '''Displacement field
        '''

        idx = self.get_time_idx(self.t_slider)
        U = self.U_t[idx]
        state = self.tstep_source.fe_domain[0]
        dof_Epia = state.xmodel.o_Epia
        fets = state.xmodel.fets
        u_Epia = U[dof_Epia]
        N_mi = fets.N_mi
        u_Emap = np.einsum('mi,Epia->Emap', N_mi, u_Epia)
        return u_Emap.reshape(-1, 2)

    def get_eps_p(self):
        '''Epsilon in the components'''
        idx = self.get_time_idx(self.t_slider)
        eps_Ems = self.get_eps_tEms(idx)
        return eps_Ems[..., (0, 2)].reshape(-1, 2)

    def get_s(self):
        '''Slip between the two material phases'''
        idx = self.get_time_idx(self.t_slider)
        eps_Ems = self.get_eps_tEms(idx)
        return eps_Ems[..., 1].flatten()

    def get_sig_p(self):
        '''Epsilon in the components'''
        idx = self.get_time_idx(self.t_slider)
        sig_Ems = self.get_sig_tEms(idx)
        return sig_Ems[..., (0, 2)].reshape(-1, 2)

    def get_sf(self):
        '''Get the shear flow in the interface
        '''
        idx = self.get_time_idx(self.t_slider)
        sig_Ems = self.get_sig_tEms(idx)
        return sig_Ems[..., 1].flatten()

    def get_eps_tEms(self, idx = slice(None)):
        '''Epsilon in the components
        '''
        txdomain = self.tstep_source.fe_domain[0]
        return txdomain.xmodel.map_U_to_field(self.U_t[idx])

    def get_sig_tEms(self, idx = slice(None)):
        '''Get stresses in the components
        '''
        reduce_dim = False
        if isinstance(idx, int):
            reduce_dim = True
            if idx == -1:
                idx_max = None
            else:
                idx_max = idx + 1
            idx = slice(idx, idx_max)
        txdomain = self.tstep_source.fe_domain[0]
        eps_tEms = self.get_eps_tEms(idx)
        t_n1 = self.t[idx]
        keys = self.Eps_t[0,0].keys()
        Eps_t = self.Eps_t[idx,0]
        if reduce_dim:
            eps_tEms = eps_tEms[0,...]
            Eps_Dt = Eps_t[0]
        else:
            Eps_Dt = {
                key: np.array([Eps[key] for i, Eps in enumerate(Eps_t)], dtype=np.float_)
                for key in keys
            }
        sig_tEms, _ = txdomain.tmodel.get_corr_pred(eps_tEms, t_n1, **Eps_Dt)
        # if reduce_dim:
        #     sig_tEms = sig_tEms[0,...]
        return sig_tEms

    def get_U_bar_t(self):
        xmodel = self.tstep_source.fe_domain[0].xmodel
        fets = xmodel.fets
        A = xmodel.A
        eps_tEms = self.get_eps_tEms(slice(0,None))
        sig_tEms = self.get_sig_tEms(slice(0,None))

        w_ip = fets.ip_weights
        J_det = xmodel.det_J_Em
        U_bar_t = 0.5 * np.einsum('m,Em,s,tEms,tEms->t',
                                  w_ip, J_det, A, sig_tEms, eps_tEms)
        return U_bar_t

    def get_W_t(self):
        P_t, _, w_L = self.get_Pw_t()
        W_t = cumtrapz(P_t, w_L, initial=0)
        return W_t

    def get_dG_t(self):
        t = self.t
        n_t = self.n_t
        U_bar_t = self.get_U_bar_t()
        W_t = self.get_W_t()
        G = W_t[:n_t] - U_bar_t[:n_t]
        if len(t) < 2:
            return np.zeros_like(t)
        tck = ip.splrep(t, G, s=0, k=1)
        return ip.splev(t, tck, der=1)

    show_legend = Bool(True, auto_set=False, enter_set=True)

    def plot_Pw(self, ax, *args, **kw):
        P_t, w_0_t, w_L_t = self.get_Pw_t()
        ymin, ymax = np.min(P_t), np.max(P_t)
        L_y = ymax - ymin
        ymax += 0.05 * L_y
        ymin -= 0.05 * L_y
        xmin, xmax = np.min(w_L_t), np.max(w_L_t)
        L_x = xmax - xmin
        xmax += 0.03 * L_x
        xmin -= 0.03 * L_x
        ax.plot(w_L_t, P_t, linewidth=2, color='black', alpha=0.4,
                label='P(w;x=L)')
        ax.plot(w_0_t, P_t, linewidth=1, color='magenta', alpha=0.4,
                label='P(w;x=0)')
        if not(ymin == ymax or xmin == xmax):
            ax.set_ylim(ymin=ymin, ymax=ymax)
            ax.set_xlim(xmin=xmin, xmax=xmax)
        ax.set_ylabel('pull-out force P [N]')
        ax.set_xlabel('pull-out slip w [mm]')
        if self.show_legend:
            ax.legend(loc=4)
        self.plot_marker(ax)

    def plot_marker(self, ax):
        P_t, w_0_t, w_L_t = self.get_Pw_t()
        idx = self.get_time_idx(self.t_slider)
        P, w = P_t[idx], w_L_t[idx]
        ax.plot([w], [P], 'o', color='black', markersize=10)
        P, w = P_t[idx], w_0_t[idx]
        ax.plot([w], [P], 'o', color='magenta', markersize=10)

    def plot_G_t(self, ax,
                 label_U='U(t)', label_W='W(t)',
                 color_U='blue', color_W='red'):

        t = self.t
        U_bar_t = self.get_U_bar_t()
        W_t = self.get_W_t()
        if len(W_t) == 0:
            return
        ax.plot(t, W_t, color=color_W, label=label_W)
        ax.plot(t, U_bar_t, color=color_U, label=label_U)
        ax.fill_between(t, W_t, U_bar_t, facecolor='gray', alpha=0.5,
                        label='G(t)')
        ax.set_ylabel('energy [Nmm]')
        ax.set_xlabel('time [-]')
        ax.legend()

    def plot_dG_t(self, ax, *args, **kw):
        t = self.t
        dG = self.get_dG_t()
        ax.plot(t, dG, color='black', label='dG/dt')
        ax.fill_between(t, 0, dG, facecolor='blue', alpha=0.05)
        ax.legend()

    # =========================================================================
    # Plot functions
    # =========================================================================
    def plot_geo(self, ax):
        u_p = self.get_u_p().T

        f_dof = self.tstep_source.free_end_dof
        w_L_b = u_p.flatten()[f_dof]
        c_dof = self.tstep_source.controlled_dof
        w = u_p.flatten()[c_dof]

        A_m = self.tstep_source.cross_section.A_m
        A_f = self.tstep_source.cross_section.A_f
        h = A_m
        d = h * 0.1  # A_f / A_m

        L_b = self.tstep_source.geometry.L_x
        x_C = np.array([[-L_b, 0], [0, 0], [0, h], [-L_b, h]], dtype=np.float_)
        ax.fill(*x_C.T, color='gray', alpha=0.3)

        f_top = h / 2 + d / 2
        f_bot = h / 2 - d / 2
        ax.set_xlim(xmin=-1.05 * L_b,
                    xmax=max(0.05 * L_b, 1.1 * self.tstep_source.w_max))

        line_F, = ax.fill([], [], color='black', alpha=0.8)
        x_F = np.array([[-L_b + w_L_b, f_bot], [w, f_bot],
                        [w, f_top], [-L_b + w_L_b, f_top]], dtype=np.float_)
        line_F.set_xy(x_F)
        x_F0 = np.array([[-L_b, f_bot], [-L_b + w_L_b, f_bot],
                         [-L_b + w_L_b, f_top], [-L_b, f_top]], dtype=np.float_)
        line_F0, = ax.fill([], [], color='white', alpha=1)
        line_F0.set_xy(x_F0)

    def plot_u_p(self, ax, label_m='matrix', label_f='reinf'):
        X_M = self.tstep_source.X_M
        L = self.tstep_source.geometry.L_x
        u_p = self.get_u_p().T
        ax.plot(X_M, u_p[0], linewidth=2, color='blue', label=label_m)
        ax.fill_between(X_M, u_p[0], 0, facecolor='blue', alpha=0.2)
        ax.plot(X_M, u_p[1], linewidth=2, color='orange', label=label_f)
        ax.fill_between(X_M, u_p[1], 0, facecolor='orange', alpha=0.2)
        ax.plot([0, L], [0, 0], color='black')
        ax.set_ylabel('displacement')
        ax.set_xlabel('bond length')
        ax.legend(loc=2)
        return np.min(u_p), np.max(u_p)

    def plot_eps_p(self, ax, label_m='matrix', label_f='reinf'):
        X_M = self.tstep_source.X_M
        L = self.tstep_source.geometry.L_x
        eps_p = self.get_eps_p().T
        ax.plot(X_M, eps_p[0], linewidth=2, color='blue', label=label_m)
        ax.fill_between(X_M, 0, eps_p[0], facecolor='blue', alpha=0.2)
        ax.plot(X_M, eps_p[1], linewidth=2, color='orange', label=label_f)
        ax.fill_between(X_M, 0, eps_p[1], facecolor='orange', alpha=0.2)
        ax.plot([0, L], [0, 0], color='black')
        ax.set_ylabel('strain')
        ax.set_xlabel('bond length')
        return np.min(eps_p), np.max(eps_p)

    def plot_sig_p(self, ax):
        X_M = self.tstep_source.X_M
        sig_p = self.get_sig_p().T
        #        A_m = self.cross_section.A_m
        #        A_f = self.cross_section.A_f
        L = self.tstep_source.geometry.L_x
        F_m = sig_p[0]
        F_f = sig_p[1]
        ax.plot(X_M, F_m, linewidth=2, color='blue', )
        ax.fill_between(X_M, F_m, 0, facecolor='blue', alpha=0.1)
        ax.plot(X_M, F_f, linewidth=2, color='orange')
        ax.fill_between(X_M, F_f, 0, facecolor='orange', alpha=0.1)
        ax.plot([0, L], [0, 0], color='black', lw=0.5)
        ax.set_ylabel('stress [MPa]')
        ax.set_xlabel('bond length')
        F_min = min(np.min(F_m), np.min(F_f))
        F_max = max(np.max(F_m), np.max(F_f))
        return F_min, F_max

    def plot_s(self, ax):
        X_J = self.tstep_source.X_M
        s = self.get_s()
        color = 'green'
        ax.fill_between(X_J, 0, s, facecolor=color, alpha=0.3)
        ax.plot(X_J, s, linewidth=2, color=color)
        ax.set_ylabel('slip')
        ax.set_xlabel('bond length')
        return np.min(s), np.max(s)

    def plot_sf(self, ax):
        X_J = self.tstep_source.X_M
        sf = self.get_sf()
        color = 'red'
        ax.fill_between(X_J, 0, sf, facecolor=color, alpha=0.2)
        ax.plot(X_J, sf, linewidth=2, color=color)
        ax.set_ylabel('shear flow')
        ax.set_xlabel('bond length')
        return np.min(sf), np.max(sf)

    t_slider = bu.Float(0)
    t_anim = bu.Float(0)
    t_max = tr.Property()
    def _get_t_max(self):
        return self.t[-1]

    ipw_view = bu.View(
        # bu.Item('t_anim', editor=bu.ProgressEditor(
        #     run_method='run',
        #     reset_method='reset',
        #     interrupt_var='interrupt',
        #     time_var='t_slider',
        #     time_max='t_max',
        # )),
        bu.Item('t_slider', editor=bu.FloatRangeEditor(
            low=0,
            high_name='t_max',
        )),
    )

    def subplots(self, fig):
        (ax_geo, ax_Pw), (ax_sf, ax_G_t) = fig.subplots(2, 2)
        ax_sig = ax_sf.twinx()
        ax_dG_t = ax_G_t.twinx()
        return ax_geo, ax_Pw, ax_sig, ax_sf, ax_G_t, ax_dG_t

    def update_plot2(self, axes):
        if len(self.U_t) == 0:
            return
        ax_geo, ax_Pw, ax_sig, ax_sf, ax_G_t, ax_dG_t = axes
        self.plot_geo(ax_geo)
        self.plot_Pw(ax_Pw)
        self.plot_sig_p(ax_sig)
        self.plot_sf(ax_sf)
        self.plot_G_t(ax_G_t)
        self.plot_dG_t(ax_dG_t)

    def update_plot(self, axes):
        if len(self.U_t) == 0:
            return
        ax_geo, ax_Pw, ax_sig, ax_sf, ax_G_t, ax_dG_t = axes
        self.plot_geo(ax_geo)
        self.plot_Pw(ax_Pw)
#        self.plot_sig_p(ax_sig)
        self.plot_s(ax_sig)
        self.plot_sf(ax_sf)
        # self.plot_G_t(ax_G_t)
        # self.plot_dG_t(ax_dG_t)
        self.tstep_source.mats_eval.bs_law.replot()
        self.tstep_source.mats_eval.bs_law.mpl_plot(ax_G_t)
        ax_G_t.set_xlabel(r'$s$ [mm]')
        ax_G_t.set_ylabel(r'$\tau$ [MPa]')
#        self.tstep_source.mats_eval.plot(ax_G_t)

class CrossSection(BMCSLeafNode, RInputRecord):
    '''Parameters of the pull-out cross section
    '''
    node_name = 'cross-section'

    A_m = bu.Float(15240,
                CS=True,
                input=True,
                unit=r'$\mathrm{mm}^2$',
                symbol=r'A_\mathrm{m}',
                auto_set=False, enter_set=True,
                desc='matrix area')
    A_f = bu.Float(153.9,
                CS=True,
                input=True,
                unit='$\\mathrm{mm}^2$',
                symbol='A_\mathrm{f}',
                auto_set=False, enter_set=True,
                desc='reinforcement area')
    P_b = bu.Float(44,
                CS=True,
                input=True,
                unit='$\\mathrm{mm}$',
                symbol='p_\mathrm{b}',
                auto_set=False, enter_set=True,
                desc='perimeter of the bond interface')

    view = View(
        Item('A_m'),
        Item('A_f'),
        Item('P_b')
    )

    tree_view = view

    ipw_view = bu.View(
        bu.Item('A_m'),
        bu.Item('A_f'),
        bu.Item('P_b')
    )

class Geometry(BMCSLeafNode, RInputRecord):
    node_name = 'geometry'
    L_x = bu.Float(45,
                GEO=True,
                input=True,
                unit='$\mathrm{mm}$',
                symbol='L',
                auto_set=False, enter_set=True,
                desc='embedded length')

    view = View(
        Item('L_x'),
    )

    tree_view = view


class DataSheet(HasStrictTraits):

    data = Array(np.float_)

    view = View(
        Item('data',
             show_label=False,
             resizable=True,
             editor=ArrayViewEditor(titles=['x', 'y', 'z'],
                                    format='%.4f',
                                    )
             ),
        width=0.5,
        height=0.6
    )


class PullOutModel(TStepBC, BMCSRootNode, Vis2D):

    name = 'Pullout'
    hist_type = PulloutHist

    node_name = 'Pull out simulation'

    tree_node_list = List([])

    def _tree_node_list_default(self):

        return [
            self.loading_scenario,
            self.mats_eval,
            self.cross_section,
            self.geometry,
#            self.sim
        ]

    def _update_node_list(self):
        self.tree_node_list = [
            self.loading_scenario,
            self.mats_eval,
            self.cross_section,
            self.geometry,
#            self.sim
        ]

    def run(self):
        self.sim.run()

    def reset(self):
        self.sim.reset()

    t = tr.Property()
    def _get_t(self):
        return self.sim.t
    def _set_t(self, value):
        self.sim.t = value

    t_max = tr.Property()
    def _get_t_max(self):
        return self.sim.t_max
    def _set_t_max(self, value):
        self.sim.t_max = value

    interrupt = tr.Property()
    def _get_interrupt(self):
        return self.sim.interrupt
    def _set_interrupt(self, value):
        self.sim.interrupt = value

    ipw_view = bu.View(
        bu.Item('t', editor=bu.ProgressEditor(
            run_method='run',
            reset_method='reset',
            interrupt_var='interrupt',
            time_var='t',
            time_max='t_max',
        )),
        bu.Item('w_max'),
        bu.Item('n_e_x'),
    )

    tree_view = View(
        Group(
            Item('mats_eval_type', resizable=True, full_size=True),
            Item('control_variable', resizable=True, full_size=True),
            Item('w_max', resizable=True, full_size=True),
            Item('n_e_x', resizable=True, full_size=True),
            Item('fixed_boundary'),
            Group(
                Item('loading_scenario@', show_label=False),
            )
        )
    )

    @tr.on_trait_change(itags_str)
    def report_change(self):
        self.model_structure_changed = True

    #=========================================================================
    # Test setup parameters
    #=========================================================================
    loading_scenario = Instance(
        LoadingScenario,
        report=True,
        desc='object defining the loading scenario'
    )

    def _loading_scenario_default(self):
        return LoadingScenario()

    cross_section = Instance(
        CrossSection,
        report=True,
        desc='cross section parameters'
    )

    def _cross_section_default(self):
        return CrossSection()

    geometry = Instance(
        Geometry,
        report=True,
        desc='geometry parameters of the boundary value problem'
    )

    def _geometry_default(self):
        return Geometry()

    control_variable = Enum('u', 'f',
                            auto_set=False, enter_set=True,
                            desc=r'displacement or force control: [u|f]',
                            BC=True)

    #=========================================================================
    # Discretization
    #=========================================================================
    n_e_x = bu.Int(20,
                    MESH=True,
                    auto_set=False,
                    enter_set=True,
                    symbol='n_\mathrm{E}',
                    unit='-',
                    desc='number of finite elements along the embedded length'
                )

    #=========================================================================
    # Algorithimc parameters
    #=========================================================================
    k_max = Int(400,
                unit='-',
                symbol='k_{\max}',
                desc='maximum number of iterations',
                ALG=True)

    tolerance = Float(1e-4,
                      unit='-',
                      symbol='\epsilon',
                      desc='required accuracy',
                      ALG=True)

    mats_eval_type = Trait('multilinear',
                           {'multilinear': MATSBondSlipMultiLinear,
                            'damage': MATSBondSlipD,
                            'elasto-plasticity': MATSBondSlipEP,
                            'damage-plasticity': MATSBondSlipDP,
                            'cumulative fatigue': MATSBondSlipFatigue},
                           MAT=True,
                           desc='material model type')

    @on_trait_change('mats_eval_type')
    def _set_mats_eval(self):
        self.mats_eval = self.mats_eval_type_()
        self._update_node_list()

    mats_eval = Instance(IMATSEval, report=True,
                         desc='material model of the interface')
    '''Material model'''

    def _mats_eval_default(self):
        return self.mats_eval_type_()

    mm = Property

    def _get_mm(self):
        return self.mats_eval

    material = Property

    def _get_material(self):
        return self.mats_eval

    #=========================================================================
    # Finite element type
    #=========================================================================
    fets_eval = Property(Instance(FETS1D52ULRH), depends_on='CS,MAT')
    '''Finite element time stepper implementing the corrector
    predictor operators at the element level'''
    @cached_property
    def _get_fets_eval(self):
        return FETS1D52ULRH(A_m=self.cross_section.A_m,
                            P_b=self.cross_section.P_b,
                            A_f=self.cross_section.A_f)

    dots_grid = Property(Instance(XDomainFEInterface1D),
                         depends_on=itags_str)
    '''Discretization object.
    '''
    @cached_property
    def _get_dots_grid(self):
        geo = self.geometry
        return XDomainFEInterface1D(
            dim_u=2,
            coord_max=[geo.L_x],
            shape=[self.n_e_x],
            fets=self.fets_eval
        )

    fe_grid = Property

    def _get_fe_grid(self):
        return self.dots_grid.mesh

    domains = Property(depends_on=itags_str + 'model_structure_changed')

    @cached_property
    def _get_domains(self):
        return [(self.dots_grid, self.mats_eval)]

    #=========================================================================
    # Boundary conditions
    #=========================================================================
    w_max = bu.Float(1, BC=True,
                  symbol='w_{\max}',
                  unit='mm',
                  desc='maximum pullout slip',
                  auto_set=False, enter_set=True)

    u_f0_max = Property(depends_on='BC')

    @cached_property
    def _get_u_f0_max(self):
        return self.w_max

    def _set_u_f0_max(self, value):
        self.w_max = value

    fixed_boundary = Enum('non-loaded end (matrix)',
                          'loaded end (matrix)',
                          'non-loaded end (reinf)',
                          'clamped left',
                          BC=True,
                          desc='which side of the specimen is fixed [non-loaded end [matrix], loaded end [matrix], non-loaded end [reinf]]')

    fixed_dofs = Property(depends_on=itags_str)

    @cached_property
    def _get_fixed_dofs(self):
        if self.fixed_boundary == 'non-loaded end (matrix)':
            return [0]
        elif self.fixed_boundary == 'non-loaded end (reinf)':
            return [1]
        elif self.fixed_boundary == 'loaded end (matrix)':
            return [self.controlled_dof - 1]
        elif self.fixed_boundary == 'clamped left':
            return [0, 1]

    controlled_dof = Property(depends_on=itags_str)

    @cached_property
    def _get_controlled_dof(self):
        return 2 + 2 * self.n_e_x - 1

    free_end_dof = Property(depends_on=itags_str)

    @cached_property
    def _get_free_end_dof(self):
        return 1

    fixed_bc_list = Property(depends_on=itags_str)
    '''Foxed boundary condition'''
    @cached_property
    def _get_fixed_bc_list(self):
        return [
            BCDof(node_name='fixed left end', var='u',
                  dof=dof, value=0.0) for dof in self.fixed_dofs
        ]

    control_bc = Property(depends_on=itags_str)
    '''Control boundary condition - make it accessible directly
    for the visualization adapter as property
    '''
    @cached_property
    def _get_control_bc(self):
        return BCDof(node_name='pull-out displacement',
                     var=self.control_variable,
                     dof=self.controlled_dof, value=self.w_max,
                     time_function=self.loading_scenario)

    bc = Property(depends_on=itags_str)

    @cached_property
    def _get_bc(self):
        return [self.control_bc] + self.fixed_bc_list

    X_M = Property()

    def _get_X_M(self):
        state = self.fe_domain[0]
        return state.xdomain.x_Ema[..., 0].flatten()

    #=========================================================================
    # Getter functions @todo move to the PulloutStateRecord
    #=========================================================================

    P = tr.Property

    def _get_P(self):
        c_dof = self.controlled_dof
        return self.F_k[c_dof]

    w_L = tr.Property

    def _get_w_L(self):
        c_dof = self.controlled_dof
        return self.U_n[c_dof]

    w_0 = tr.Property

    def _get_w_0(self):
        f_dof = self.free_end_dof
        return self.U_n[f_dof]

    def get_shear_integ(self):
        sf_t_Em = np.array(self.tloop.sf_Em_record)
        w_ip = self.fets_eval.ip_weights
        J_det = self.tstepper.J_det
        P_b = self.cross_section.P_b
        shear_integ = np.einsum('tEm,m,em->t', sf_t_Em, w_ip, J_det) * P_b
        return shear_integ

    def plot_omega(self, ax, vot):
        X_J = self.X_J
        omega = self.get_omega(vot)
        ax.fill_between(X_J, 0, omega, facecolor='lightcoral', alpha=0.3)
        ax.plot(X_J, omega, linewidth=2, color='lightcoral', label='bond')
        ax.set_ylabel('damage')
        ax.set_xlabel('bond length')
        ax.legend(loc=2)
        return 0.0, 1.05

    def plot_eps_s(self, ax, vot):
        eps_p = self.get_eps_p(vot).T
        s = self.get_s(vot)
        ax.plot(eps_p[1], s, linewidth=2, color='lightcoral')
        ax.set_ylabel('reinforcement strain')
        ax.set_xlabel('slip')

    def subplots(self, fig):
        ax_geo, ax_Pw = fig.subplots(1,2)
        return ax_geo, ax_Pw

    def update_plot(self, axes):
        if len(self.hist.U_t) == 0:
            return
        ax_geo, ax_Pw = axes
        self.hist.plot_geo(ax_geo)
        self.hist.plot_Pw(ax_Pw)

    def get_window(self):
        Pw = self.hist.plt('plot_Pw', label='pullout curve')
        geo = self.plt('plot_geo', label='geometry')
        u_p = self.plt('plot_u_p', label='displacement along the bond')
        eps_p = self.plt('plot_eps_p', label='strain along the bond')
        sig_p = self.plt('plot_sig_p', label='stress along the bond')
        s = self.plt('plot_s', label='slip along the bond')
        sf = self.plt('plot_sf', label='shear flow along the bond')
        energy = self.hist.plt('plot_G_t', label='energy')
        dissipation = self.hist.plt('plot_dG_t', label='energy release')
        pp0 = PlotPerspective(
            name='geo',
            viz2d_list=[geo],
            positions=[111],
        )
        pp1 = PlotPerspective(
            name='history',
            viz2d_list=[Pw, geo, energy, dissipation],
            positions=[221, 222, 223, 224],
        )
        pp2 = PlotPerspective(
            name='fields',
            viz2d_list=[s, u_p, eps_p, sig_p],
            twinx=[(s, sf, False)],
            positions=[221, 222, 223, 224],
        )
        win = BMCSWindow(model=self)
        win.viz_sheet.pp_list = [pp0, pp1, pp2]
        win.viz_sheet.selected_pp = pp0
        win.viz_sheet.monitor_chunk_size = 10
        return win

