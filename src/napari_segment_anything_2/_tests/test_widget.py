from typing import Callable

import napari
import numpy as np
import pytest
from numpy.typing import ArrayLike
from skimage.data import astronaut, camera

from napari_segment_anything_2 import SAM2Widget


@pytest.mark.parametrize("image", [astronaut(), camera()])
def test_automatic_segmentation(
    make_napari_viewer: Callable[[], napari.Viewer],
    image: ArrayLike,
    request,
) -> None:
    viewer = make_napari_viewer()
    widget = SAM2Widget(viewer, model_type="sam2_hiera_t")

    viewer.window.add_dock_widget(widget)
    video = np.stack([image] * 3)
    viewer.add_image(video)

    widget._run_btn.clicked()

    if request.config.getoption("--show-napari-viewer"):
        napari.run()
        return

    labels = viewer.layers[-1].data
    assert np.any(labels > 0)
