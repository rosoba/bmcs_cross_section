import traits.api as tr
from bmcs_utils.api import InteractiveModel
import numpy as np
import sympy as sp
from bmcs_utils.api import View, Item, Float
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.patches import Circle as MPL_Circle


class ICrossSectionShape(tr.Interface):
    """This interface lists the functions need to be implemented by cross section classes."""

    def get_cs_area(self):
        """Returns the area of the cross section."""

    def get_cs_i(self):
        """Returns the moment of inertia of the cross section
         with respect to an axis passing through its centroid ."""

    def get_b(self, z_positions_array):
        """Returns b values that correspond to the positions z_positions_array."""


class CrossSectionShapeBase(InteractiveModel):
    """"This class describes the geometry of the cross section."""
    name = 'Cross Section Shape'

    H = Float(400, GEO=True)

    ipw_view = View(
        Item('H', minmax=(10, 3000), latex='H [mm]')
    )


@tr.provides(ICrossSectionShape)
class Rectangle(CrossSectionShapeBase):

    B = Float(200, GEO=True)

    ipw_view = View(
        *CrossSectionShapeBase.ipw_view.content,        # this will add View Items of the base class CrossSectionShapeBase
        Item('B', minmax=(10, 500), latex='B [mm]')
    )

    def get_cs_area(self):
        return self.B * self.H

    def get_cs_i(self):
        return self.B * self.H ** 3 / 12

    def get_b(self, z_positions_array):
        return np.full_like(z_positions_array, self.B)

    def update_plot(self, ax):

        ax.fill([-self.B/2, self.B/2, self.B/2, -self.B/2, -self.B/2], [0, 0, self.H, self.H, 0], color='gray', alpha=0.2 )
        ax.plot([-self.B/2, self.B/2, self.B/2, -self.B/2, -self.B/2], [0, 0, self.H, self.H, 0], color='black')
        ax.autoscale_view()
        ax.set_aspect('equal')

        ax.annotate('Area = {} $mm^2$'.format(int(Rectangle.get_cs_area(self)), 0),
                    xy=(-self.H/2 * 0.8, self.H/2), color='blue')
        ax.annotate('I = {} $mm^4$'.format(int(Rectangle.get_cs_i(self)), 0),
                    xy=(-self.H/2 * 0.8, self.H/2 * 0.8), color='blue')


@tr.provides(ICrossSectionShape)
class Circle(CrossSectionShapeBase):

    # TODO->Rostia: provide input field instead minmax range
    # H from the base class is used as the D, for the diameter of the circular section

    H = Float(250, GEO=True)

    ipw_view = View(
        Item('H', minmax=(10, 3000), latex='D [mm]'),
    )

    def get_cs_area(self):
        return np.pi * self.H * self.H

    def get_cs_i(self):
        return np.pi * self.H ** 4 / 4

    def get_b(self, z_positions_array):
        return np.full_like(z_positions_array, self.H)

        # TODO->Saeed: complete this


    def update_plot(self, ax):
        # TODO->Saeed: fix this

        circle = MPL_Circle((0, self.H/2), self.H/2, facecolor=(.5,.5,.5,0.2), edgecolor=(0,0,0,1))

        ax.add_patch(circle)
        ax.autoscale_view()
        ax.set_aspect('equal')
        ax.annotate('Area = {} $mm^2$'.format(int(self.get_cs_area()), 0),
                    xy=(-self.H/2 * 0.8, self.H/2), color='blue')
        ax.annotate('I = {} $mm^4$'.format(int(self.get_cs_i()), 0),
                    xy=(-self.H/2 * 0.8, self.H/2 * 0.8), color='blue')


@tr.provides(ICrossSectionShape)
class TShape(CrossSectionShapeBase):

    B_f = Float(200, GEO=True)
    B_w = Float(50, GEO=True)
    H_w = Float(150, GEO=True)

    ipw_view = View(
        *CrossSectionShapeBase.ipw_view.content,
        Item('B_f', minmax=(10, 3000), latex=r'B_f \mathrm{[mm]}'),
        Item('B_w', minmax=(10, 3000), latex=r'B_w \mathrm{[mm]}'),
        Item('H_w', minmax=(10, 3000), latex=r'H_w \mathrm{[mm]}'),
    )

    def get_cs_area(self):
        return self.B_w * self.H_w + self.B_f * (self.H - self.H_w)

    def get_cs_i(self):
        A_f = self.B_f * (self.H - self.H_w)
        Y_f = (self.H - self.H_w) / 2 + self.H_w
        I_f = self.B_f * (self.H - self.H_w) ** 3 / 12
        A_w = self. B_w * self.H_w
        Y_w = self.H_w / 2
        I_w = self.B_w * self.H_w ** 3 /12
        Y_c = (Y_f * A_f + Y_w * A_w) / (A_f + A_w)
        I_c = I_f + A_f * (Y_c - Y_f)**2 + I_w + A_w * (Y_c - Y_w)**2
        return Y_c, I_c


    get_b = tr.Property(tr.Callable, depends_on='+input')

    @tr.cached_property
    def _get_get_b(self):
        z_ = sp.Symbol('z')
        b_p = sp.Piecewise(
            (self.B_w, z_ < self.H_w),
            (self.B_f, True)
        )
        return sp.lambdify(z_, b_p, 'numpy')

    def update_plot(self, ax):
        # TODO [Saeed]: fix this
        # Start drawing from bottom center of the cross section
        cs_points = np.array([  [self.B_w/2, 0],
                                [self.B_w/2, self.H_w],
                                [self.B_f/2, self.H_w],
                                [self.B_f/2, self.H],
                                [-self.B_f/2, self.H],
                                [-self.B_f/2, self.H_w],
                                [-self.B_w/2, self.H_w],
                                [-self.B_w/2, 0]])

        cs = Polygon(cs_points)

        patch_collection = PatchCollection([cs], facecolor=(.5,.5,.5,0.2), edgecolors=(0,0,0,1))

        ax.add_collection(patch_collection)
        ax.scatter(0, TShape.get_cs_i(self)[0], color ='white', s = self.B_w, marker ="+")

        ax.autoscale_view()
        ax.set_aspect('equal')

        ax.annotate('Area = {} $mm^2$'.format(int(TShape.get_cs_area(self)), 0),
                    xy=(-self.H/2 * 0.8, (self.H / 2 + self.H_w / 2)), color='blue')
        ax.annotate('I = {} $mm^4$'.format(int(TShape.get_cs_i(self)[1]), 0),
                    xy=(-self.H/2 * 0.8, (self.H / 2 + self.H_w / 2) * 0.9), color='blue')
        ax.annotate('$Y_c$',
                    xy=(0, TShape.get_cs_i(self)[0] * 0.85), color='purple')

        # ax.annotate('$B_w$', xy=(-self.B_w / 2 , -self.B_w * 0.1), xytext=(self.B_w / 2 , -self.B_w * 0.1),
        #             arrowprops={'arrowstyle': '<->'}, va='center', ha ='center')
        # ax.annotate('$B_f$', xy=(-self.B_f / 2 , self.H * 1.1), xytext=(self.B_f / 2 , self.H * 1.1),
        #             arrowprops={'arrowstyle': '<->'}, va='center', ha ='center')
        # ax.annotate('$H$', xy=(-self.B_f / 2 , self.H * 1.1), xytext=(-self.B_f / 2 , self.H * 1.1 ),
        #             arrowprops={'arrowstyle': '<->'}, va='center', ha ='center')
        # ax.annotate('$H_w$', xy=(-self.B_f / 2 * 1.1, 0), xytext=(-self.B_f / 2 * 1.1 , self.H_w),
        #             arrowprops={'arrowstyle': '<->'}, va='center', ha ='center')


# TODO->Saeed: maybe complete this
@tr.provides(ICrossSectionShape)
class CustomShape(CrossSectionShapeBase):

    def get_cs_area(self):
        pass

    def get_cs_i(self):
        pass

    def get_b(self, z_positions_array):
        pass

    def update_plot(self, ax):
        # TODO->Saeed: fix this to use it for the CustomShape
        # self.update_plot(ax)
        z = np.linspace(0, self.H, 100)
        b = self.get_b(z)
        ax.axis([0, np.max(b), 0, self.H])
        ax.axis('equal')
        ax.fill(b, z, color='gray')
        ax.plot(b, z, color='black')
        ax.autoscale_view()
        ax.set_aspect('equal')
        ax.annotate('Area = {} $mm^2$'.format(int(CustomShape.get_cs_area(self)), 0),
                    xy=(-self.H/2 * 0.8, self.H/2), color='blue')
        ax.annotate('I = {} $mm^4$'.format(int(CustomShape.get_cs_i(self)), 0),
                    xy=(-self.H/2 * 0.8, self.H/2 * 0.8), color='blue')