import numpy as np
import traits.api as tr
from bmcs_utils.api import \
    InteractiveModel, View
from .cs_reinf_layer import ReinfLayer

from bmcs_cross_section.matmod import \
    ConcreteMatMod, PWLConcreteMatMod, EC2ConcreteMatMod

class CrossSectionLayout(InteractiveModel):
    name = 'Cross Section Layout'

    cs_design = tr.WeakRef

    matrix = tr.Instance(ConcreteMatMod)
    def _matrix_default(self):
        return PWLConcreteMatMod()

    reinforcement = tr.List([])

    def add_layer(self, rl):
        rl.cs_layout = self
        self.reinforcement.append(rl)

    z_j = tr.Property
    def _get_z_j(self):
        return np.array([r.z for r in self.reinforcement], dtype=np.float_)

    A_j = tr.Property
    def _get_A_j(self):
        return np.array([r.A for r in self.reinforcement], dtype=np.float_)

    def get_N_tj(self, eps_tj):
        return np.array([r.get_N(eps_t)
                         for r, eps_t in zip(self.reinforcement, eps_tj.T)],
                         dtype=np.float_).T

    ipw_view = View(
    )

    def subplots(self, fig):
        return fig.subplots(1, 1)

    def update_plot(self, ax):
        self.cs_design.cross_section_shape.update_plot(ax)

        H = int(self.cs_design.cross_section_shape.H)

        maxA = 0
        for reinforcement in self.reinforcement:
            A = reinforcement.A
            maxA = max(A, maxA)

        for reinforcement in self.reinforcement:
            z = reinforcement.z
            A = reinforcement.A
            b = self.cs_design.cross_section_shape.get_b([z])
            ax.plot([-0.9 * b/2, 0.9 * b/2], [z, z], color='r',
                    linewidth=5 * A/maxA)

        # ax.annotate(
        #     'E_composite = {} GPa'.format(np.round(self.get_comp_E() / 1000), 0),
        #     xy=(-H / 2 * 0.8, (H / 2 + H / 2) * 0.8), color='blue'
        # )

