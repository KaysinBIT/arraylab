"""Paired sparse figures keep one scientific selection and readable annotations."""
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
from matplotlib.backend_bases import ResizeEvent
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from arraylab.sparse2d.core import Config, candidates, evaluate_result
from arraylab.sparse2d.elements import make_snapshot
from arraylab.sparse2d.plots import draw_detail_plot, draw_layout_pattern
from arraylab.sparse2d.scenarios import scenarios


@pytest.fixture(scope='module')
def sample():
    config = Config(nx=4,ny=4,k=8,opt_grid=25,eval_grid=81,phase_bits=2,
                    element_model='patch',scan_angles='0,0;27,34')
    integer,xy,_ = candidates(config)
    indices = np.array([0,2,5,7,8,10,13,15])
    result = {'xy':xy[indices],'integer':integer[indices],'indices':indices,
              'weights':np.array([1,2,3,4,4,3,2,1],dtype=float)/20,
              'name':'Greedy exchange'}
    evaluate_result(result,config)
    bundle = {'config':asdict(config),'element_pattern':make_snapshot(config),
              'scenarios':scenarios(config),'results':[result]}
    return bundle,result


@pytest.mark.parametrize('view,phase', [('total','ideal'),('total','quantized'),
                                       ('array_factor','quantized'),('element','ideal')])
def test_paired_geometry_and_field_match_selected_snapshot(sample,view,phase):
    bundle,result = sample
    before = deepcopy(result)
    paired = Figure(figsize=(8,5),dpi=100)
    FigureCanvasAgg(paired)
    axes = draw_layout_pattern(paired,bundle,result,1,view,phase)
    active = next(item for item in axes[0].collections if item.get_label()=='Active')
    np.testing.assert_array_equal(active.get_offsets(),result['xy'])
    assert np.argmax(active.get_sizes()) == np.argmax(result['weights'])
    reference = Figure(figsize=(8,5),dpi=100)
    FigureCanvasAgg(reference)
    field = draw_detail_plot(reference,bundle,result,'pattern',1,view,phase_mode=phase)
    np.testing.assert_array_equal(axes[1].images[0].get_array(),field.images[0].get_array())
    np.testing.assert_array_equal(axes[1].lines[0].get_xydata(),field.lines[0].get_xydata())
    assert paired._arraylab_footer_lines == reference._arraylab_footer_lines
    for key in ('xy','weights','integer','indices'):
        np.testing.assert_array_equal(result[key],before[key])
    assert result['scenario_metrics'] == before['scenario_metrics']


@pytest.mark.parametrize('width,height,dpi', [(5,3.6,100),(7,4.5,125),(10,7.2,100)])
def test_paired_axes_footer_and_labels_fit_without_overlap(sample,width,height,dpi):
    bundle,result = sample
    figure = Figure(figsize=(width,height),dpi=dpi)
    canvas = FigureCanvasAgg(figure)
    axes = draw_layout_pattern(figure,bundle,result,1,'total','quantized')
    canvas.draw()
    renderer = canvas.get_renderer()
    assert axes[0].get_position().x1 < axes[1].get_position().x0
    footer = figure._arraylab_metric_footer.get_window_extent(renderer)
    assert footer.x0 >= 0 and footer.x1 <= figure.bbox.width
    boxes = [ax.get_tightbbox(renderer) for ax in figure.axes]
    for index,bounds in enumerate(boxes):
        assert not bounds.overlaps(footer)
        assert bounds.x0 >= -1 and bounds.x1 <= figure.bbox.width+1
        assert bounds.y0 >= -1 and bounds.y1 <= figure.bbox.height+1
        assert not any(bounds.overlaps(other) for other in boxes[index+1:])


def test_resize_reflows_existing_field_without_resampling(sample,monkeypatch):
    import arraylab.sparse2d.plots as plots
    bundle,result = sample
    figure = Figure(figsize=(8,5),dpi=100)
    canvas = FigureCanvasAgg(figure)
    axes = draw_layout_pattern(figure,bundle,result,1,'total','quantized')
    field = axes[1].images[0]
    original = field.get_array().copy()
    monkeypatch.setattr(plots,'_plot_response',lambda *args: pytest.fail('resize resampled field'))
    monkeypatch.setattr(plots,'_metric_footer',lambda *args: pytest.fail('resize recalculated metrics'))
    for width,height in ((5,3.6),(10,7.2)):
        figure.set_size_inches(width,height,forward=False)
        ResizeEvent('resize_event',canvas)._process()
        canvas.draw()
        assert axes[1].images[0] is field
        np.testing.assert_array_equal(field.get_array(),original)
        assert axes[0].get_position().x1 < axes[1].get_position().x0
