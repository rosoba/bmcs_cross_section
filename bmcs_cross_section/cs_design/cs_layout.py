import numpy as np
import traits.api as tr
from bmcs_utils.api import \
    InteractiveModel, Item, View, Float, Int


class Reinforcement(InteractiveModel):
    name = 'Reinforcement'

    # TODO->Saeed: prepare the variables for InteractiveModel (ipw_view and so on...)

    z_j = tr.Array(np.float_, value=[50])
    """z positions of reinforcement layers"""

    A_j = tr.Array(np.float_, value=[3 * np.pi * (16 / 2.) ** 2])
    """cross section area of reinforcement layers"""

    E_j = tr.Array(np.float_, value=[210000])
    """E modulus of reinforcement layers"""

    eps_sy_j = tr.Array(np.float_, value=[500. / 210000.])
    """Steel yield strain"""

    ipw_view = View(

        Item('z_j', latex='z_{j} \mathrm{[mm]}'),
        Item('x_j', latex='x_{j} \mathrm{[mm]}'),
        Item('A_j', latex='A_{j} \mathrm{[mm^2]}'),
        Item('E_j', latex='E_{j} \mathrm{[MPa]}'),
        Item('eps_sy_j', latex='eps_{sy_j} \mathrm{[-]}'),
    )

class Fabric(Reinforcement):
    """Reinforcement with a grid structure
    """
    E_carbon = Int(200000)
    width = Float(8)
    thickness = Float(1)
    spacing = Float(1)
    n_layers = Int(1)
    A_roving = Float(1)
    f_h = Int(5)

    ipw_view = View(
        Item('E_carbon', latex='E_r \mathrm{[MPa]}', minmax=(200000, 300000)),
        Item('width', latex='rov_w \mathrm{[mm]}', minmax=(8, 450)),
        Item('thickness', latex='rov_t \mathrm{[mm]}', minmax=(1, 100)),
        Item('spacing', latex='ro_s \mathrm{[mm]}', minmax=(1, 100)),
        Item('n_layers', latex='n_l \mathrm{[-]}', minmax=(1, 100)),
        Item('A_roving', latex='A_r \mathrm{[mm^2]}', minmax=(1, 100)),
        Item('f_h', latex='f_h \mathrm{[mm]}', minmax=(5, 500))
    )


class Bar(Reinforcement):
    """" pass """

class Matrix(InteractiveModel):
    name = 'Matrix'

    E_ct = Float(24000)
    """E modulus of matrix on tension"""

    E_cc = Float(25000)
    """E modulus of matrix on compression"""

    eps_cr = Float(0.001)
    """Matrix cracking strain"""

    eps_cy = Float(-0.003)
    """Matrix compressive yield strain"""

    eps_cu = Float(-0.01)
    """Ultimate matrix compressive strain"""

    eps_tu = Float(0.003)
    """Ultimate matrix tensile strain"""

    mu = Float(0.33)
    """Post crack tensile strength ratio (represents how much strength is left after the crack because of short steel 
    fibers in the mixture)"""

    ipw_view = View(
        Item('E_ct', latex='E_{ct} [N/mm^2]'),
        Item('E_cc', latex='E_{cc} [N/mm^2]'),
        Item('eps_cr'),
        Item('eps_cy'),
        Item('eps_cu'),
        Item('eps_tu'),
        Item('mu')
    )

    def update_plot(self, axes):
        pass


class CrossSectionLayout(InteractiveModel):
    name = 'CrossSectionLayout'

    cs_design = tr.WeakRef


    # print = Info.trait_get(Info.trait_names())

        # = self.trait_names()
        # print(self.trait_get(trait_names))

    matrix = tr.Instance(Matrix, ())
    reinforcement = tr.Instance(Reinforcement, ())
    fabric = tr.Instance(Fabric, ())
    bar = tr.Instance(Bar, ())

    def get_shape(self):
        rec = tr.Instance(self.cs_design.cross_section_shape, ())

    def get_reinf(self):
        pass

    def get_comp_E(self):

        # give him a B value
        B = 100
        H = self.cs_design.cross_section_shape.H
        # A_composite = B * H
        A_composite = B * H

        n_rovings = self.fabric.width / self.fabric.spacing  # width or B??
        A_layer = n_rovings * self.fabric.A_roving
        A_carbon = self.fabric.n_layers * A_layer
        A_concrete = A_composite - A_carbon
        E_comp = (self.fabric.E_carbon * A_carbon + self.matrix.E_cc * A_concrete) / (A_composite)
        return E_comp

    ipw_view = View(
    )

    def subplots(self, fig):
        return fig.subplots(1, 1)

    def update_plot(self, ax):
        self.cs_design.cross_section_shape.update_plot(ax)

        # TODO->Saeed: the previous line will plot a cross section, please add the steel to it just as red strips in z_j locations
        #  and with a width that is relative to A_j (get z_j and A_j values from 'reinforcement' class variable)
        #  (just fix, generalize and improve the following)
        H = int(self.cs_design.cross_section_shape.H)

        max_B = np.max(self.cs_design.cross_section_shape.get_b(np.linspace(0, 100, H)))
        z1 = self.reinforcement.z_j[0]
        ax.plot([-max_B/2, max_B/2], [z1, z1], color='r', linewidth=5)


        # ax.plot([self.b / 2 - self.width / 2, self.b / 2 + self.width / 2], [self.f_h, self.f_h], color='Blue',
        #         linewidth=self.n_layers * self.thickness)
        ax.annotate('E_composite = {} GPa'.format(np.round(self.get_comp_E() / 1000), 0),
                    xy=(-H / 2 * 0.8, (H / 2 + H / 2) * 0.8), color='blue')

