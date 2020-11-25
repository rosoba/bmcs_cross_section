from .cs_layout import CrossSectionLayout
from .cs_shape import RectangleCS, CircleCS, TShapeCS, CustomShapeCS, ICrossSectionShape
from bmcs_utils.api import InteractiveModel, Item, View
import traits.api as tr
from bmcs_utils.trait_types import Float, Bool, Int, FloatRangeEditor


class CSDesign(InteractiveModel):
    name = 'Cross Section Design'

    cross_section_layout = tr.Instance(CrossSectionLayout)

    def _cross_section_layout_default(self):
        return CrossSectionLayout(beam_design=self)

    H = tr.DelegatesTo('cross_section_shape')

    # Cross section shape options
    Rectangle = Bool(True)
    Circle = Bool(False)
    TShape = Bool(False)
    CustomShape = Bool(False)

    # Reinforement options
    Fabric = Bool(False)
    Bar = Bool(False)

    cross_section_shape = tr.Instance(ICrossSectionShape)
    def _cross_section_shape_default(self):
#         if self.Rectangle == True:
#             shape = RectangleCS
#         elif self.Circle == True:
#             shape = CircleCS
#         elif self.TShape == True:
#             shape = TShapeCS
#         elif self.CustomShape == True:
#             shape = CustomShapeCS
        # TODO [Homam] fix this
        return RectangleCS()

    ipw_view = View(
        Item('Rectangle', latex=r'Rectangle'),
        Item('Circle', latex=r'Circle'),
        Item('TShape', latex=r'TShape'),
        Item('CustomShape', latex=r'CustomShape'),
        Item('Fabric', latex='Fabric'),
        Item('Bar', latex=r'Bar')
    )

    def subplots(self, fig):
        return fig.subplots(1, 1)

    def update_plot(self, ax):
        self.cross_section_shape.update_plot(ax)
        self.cross_section_layout.update_plot(ax)
