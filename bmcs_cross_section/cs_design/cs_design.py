
from .cs_layout import CrossSectionLayout
from .cs_shape import RectangleCS, ICrossSectionShape
from bmcs_utils.api import InteractiveModel, Item, View
import traits.api as tr


class CSDesign(InteractiveModel):
    cross_section_layout = tr.Instance(CrossSectionLayout)

    def _cross_section_layout_default(self):
        return CrossSectionLayout(beam_design=self)

    cross_section_shape = tr.Instance(ICrossSectionShape)

    def _cross_section_shape_default(self):
        return RectangleCS()

    name = 'Cross section'

    H = tr.DelegatesTo('cross_section_shape')

    ipw_view = View(
    )

    def subplots(self, fig):
        return fig.subplots(1, 1)

    def update_plot(self, ax):
        self.cross_section_shape.update_plot(ax)
        self.cross_section_layout.update_plot(ax)