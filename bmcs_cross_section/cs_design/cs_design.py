from .cs_layout import CrossSectionLayout
from .cs_shape import Rectangle, Circle, TShape, CustomShape, ICrossSectionShape
from bmcs_utils.api import Model, Item, View
import traits.api as tr
from bmcs_utils.trait_types import \
    Float, Bool, Int, FloatRangeEditor, EitherType, Instance
from bmcs_cross_section.matmod import \
    PWLConcreteMatMod, EC2ConcreteMatMod



class CrossSectionDesign(Model):
    name = 'Cross Section Design'

    matrix = EitherType(options=[
        ('piecewise linear', PWLConcreteMatMod),
        ('EC2 with plateau', EC2ConcreteMatMod)
        ], MAT=True)


    cross_section_layout = Instance(CrossSectionLayout)

    def _cross_section_layout_default(self):
        return CrossSectionLayout(cs_design=self)

    tree = ['matrix','cross_section_layout','cross_section_shape']

    csl = tr.Property
    def _get_csl(self):
        return self.cross_section_layout

    H = tr.DelegatesTo('cross_section_shape_')

    cross_section_shape = EitherType(
                          options=[('rectangle', Rectangle),
                                    ('circle', Circle),
                                    ('T-shape', TShape),
                                   ('custom', CustomShape)],
                          CS=True, tree=True )

    ipw_view = View(
        Item('matrix', latex=r'\mathrm{concrete behavior}'),
        Item('cross_section_shape', latex=r'\mathrm{shape}'),
        Item('cross_section_layout', latex=r'\mathrm{layout}'),
    )

    def subplots(self, fig):
        return fig.subplots(1, 1)

    def update_plot(self, ax):
        self.cross_section_shape_.update_plot(ax)
        self.cross_section_layout.update_plot(ax)
