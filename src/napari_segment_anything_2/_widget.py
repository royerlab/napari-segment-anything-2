from typing import Any, Generator, Optional, Tuple

import napari
import numpy as np
import pandas as pd
import torch
from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    PushButton,
    create_widget,
)
from napari.layers import Image, Points, Shapes
from napari.layers.shapes._shapes_constants import Mode
from qtpy.QtCore import Qt
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage import color, measure, transform, util

from napari_segment_anything_2.sam2 import (
    SAM2VideoPredictor,
    build_sam2_video_predictor,
)
from napari_segment_anything_2.utils import (
    SAM_WEIGHTS_URL,
    get_weights_path,
    wait_cursor,
)

GlobalHydra.instance().clear()
initialize_config_module(
    "napari_segment_anything_2/configs", version_base="1.3"
)


class SAM2Widget(Container):
    _predictor: SAM2VideoPredictor
    _image_predictor: SAM2ImagePredictor

    def __init__(
        self, viewer: napari.Viewer, model_type: str = "sam2_hiera_b+"
    ):
        super().__init__()
        self._viewer = viewer
        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        self._score_threshold = 0.0

        self._model_type_widget = ComboBox(
            value=model_type,
            choices=list(SAM_WEIGHTS_URL.keys()),
            label="Model:",
        )
        self._model_type_widget.changed.connect(self._load_model)
        self.append(self._model_type_widget)

        self._im_layer_widget = create_widget(annotation=Image, label="Image:")
        self._im_layer_widget.changed.connect(self._load_image)
        self.append(self._im_layer_widget)

        self._confirm_mask_btn = PushButton(
            text="Confirm Annot.",
            enabled=False,
            tooltip="Press C to confirm annotation.",
        )
        self._confirm_mask_btn.changed.connect(self._on_confirm_mask)
        self.append(self._confirm_mask_btn)

        self._cancel_annot_btn = PushButton(
            text="Cancel Annot.",
            enabled=False,
            tooltip="Press X to cancel annotation.",
        )
        self._cancel_annot_btn.changed.connect(self._cancel_annot)
        self.append(self._cancel_annot_btn)

        self._auto_segm_btn = PushButton(text="Auto. Segm.")
        self._auto_segm_btn.changed.connect(self._on_auto_segm)
        self.append(self._auto_segm_btn)

        self._tracks_ckbox = CheckBox(text="Show tracks")
        self._tracks_ckbox.value = False
        self.append(self._tracks_ckbox)

        self._labels_layer = self._viewer.add_labels(
            data=np.zeros((1, 256, 256), dtype=int),
            name="SAM labels",
        )

        self._mask_layer = self._viewer.add_labels(
            data=np.zeros((1, 256, 256), dtype=int),
            name="SAM mask",
            colormap={1: "cyan", None: "transparent"},
        )
        self._mask_layer.contour = 2

        self._pts_layer = self._viewer.add_points(name="SAM points")
        self._pts_layer.current_face_color = "blue"
        self._pts_layer.events.data.connect(self._on_interactive_run)
        self._pts_layer.mouse_drag_callbacks.append(
            self._mouse_button_modifier
        )
        self._boxes_layer = self._viewer.add_shapes(
            name="SAM box",
            face_color="transparent",
            edge_color="green",
            edge_width=2,
        )
        self._boxes_layer.mouse_drag_callbacks.append(self._on_shape_drag)

        self._init_frame: Optional[np.ndarray] = None
        self._video: Optional[np.ndarray] = None
        self._logits: Optional[torch.TensorType] = None

        self._model_type_widget.changed.emit(model_type)
        self._im_layer_widget.changed.emit(self._im_layer_widget.value)

        self._viewer.bind_key("C", self._on_confirm_mask)
        self._viewer.bind_key("X", self._cancel_annot)

    def _load_model(self, model_type: str) -> None:
        self._predictor = build_sam2_video_predictor(
            model_type, get_weights_path(model_type)
        )
        self._predictor.fill_hole_area = 0
        self._mask_generator = SAM2AutomaticMaskGenerator(self._predictor)
        self._image_predictor = SAM2ImagePredictor(self._predictor)

    @wait_cursor()
    def _load_image(self, im_layer: Optional[Image]) -> None:
        is_valid = False
        self._init_frame, self._video = None, None
        if im_layer is not None:
            image = im_layer.data
            if im_layer.ndim < 2 or im_layer.ndim > 3:
                print(
                    "Image must have 2 (YX) or 3 (TYX) plus optional RGB channels. "
                    f"Got {image.ndim}"
                )
            else:
                is_valid = True
                self._init_frame, self._video = self._load_video()

        self._auto_segm_btn.enabled = is_valid

    def _mouse_button_modifier(self, _: Points, event) -> None:
        self._pts_layer.selected_data = []
        if event.button == Qt.LeftButton:
            self._pts_layer.current_face_color = "blue"
        else:
            self._pts_layer.current_face_color = "red"

    def _on_interactive_run(self, _: Optional[Any] = None) -> None:
        points = self._pts_layer.data
        boxes = self._boxes_layer.data

        if self._im_layer_widget.value is None or (
            len(points) == 0 and len(boxes) == 0
        ):
            return

        if len(boxes) > 0:
            box = boxes[-1]
            box = np.stack([box.min(axis=0), box.max(axis=0)], axis=0)
            box = np.flip(box, -1).reshape(-1)[None, ...]
        else:
            box = None

        if len(points) > 0:
            points = np.flip(points, axis=-1).copy()
            colors = self._pts_layer.face_color
            blue = [0, 0, 1, 1]
            labels = np.all(colors == blue, axis=1)
        else:
            points = None
            labels = None

        mask, _, self._logits = self._image_predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=box,
            mask_input=self._logits,
            multimask_output=False,
        )
        self._mask_layer.data = mask.astype(int)
        self._confirm_mask_btn.enabled = True
        self._cancel_annot_btn.enabled = True

    def _on_shape_drag(self, _: Shapes, event) -> Generator:
        if self._boxes_layer.mode != Mode.ADD_RECTANGLE:
            return
        # on mouse click
        yield
        # on move
        while event.type == "mouse_move":
            yield
        # on mouse release
        self._on_interactive_run()

    @wait_cursor()
    def _on_auto_segm(self) -> None:
        if self._im_layer_widget.value is None:
            return

        video_masks = np.zeros(
            (self._video.shape[0], self._video.shape[2], self._video.shape[3]),
            dtype=np.int32,
        )

        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            state = self._predictor.init_state(self._video)
            self._predictor.reset_state(state)

            masks = self._mask_generator.generate(self._init_frame)
            for i, item in enumerate(masks):
                _, _, masks = self._predictor.add_new_mask(
                    state,
                    frame_idx=0,
                    obj_id=i + 1,
                    mask=item["segmentation"],
                )

            for (
                frame_idx,
                obj_ids,
                mask_logits,
            ) in self._predictor.propagate_in_video(state):
                for i, obj_idx in enumerate(obj_ids):
                    mask = (
                        (mask_logits[i] > self._score_threshold)[0]
                        .cpu()
                        .numpy()
                    )
                    video_masks[frame_idx][mask] = obj_idx

        video_masks = np.flip(video_masks, axis=0)
        video_masks = transform.resize(
            video_masks,
            self._im_layer_widget.value.data.shape[:3],
            order=0,
        )

        self._labels_layer.data = video_masks

        if self._tracks_ckbox.value:
            # TODO: make available in interactive mode
            df = []
            for t, frame in enumerate(video_masks):
                _df = pd.DataFrame(
                    measure.regionprops_table(
                        frame, properties=("label", "centroid")
                    )
                )
                _df["t"] = t
                df.append(_df)
            df = pd.concat(df)
            self._viewer.add_tracks(
                df[["label", "t", "centroid-0", "centroid-1"]],
                name="Tracks",
                colormap="hsv",
                blending="translucent",
            )

    def _load_video(self) -> Tuple[np.ndarray, torch.Tensor]:
        if self._im_layer_widget.value is None:
            raise ValueError("Image layer is not set.")

        img_mean = torch.tensor(
            (0.485, 0.456, 0.406), dtype=torch.float32, device=self._device
        )[:, None, None]
        img_std = torch.tensor(
            (0.229, 0.224, 0.225), dtype=torch.float32, device=self._device
        )[:, None, None]

        image = self._im_layer_widget.value.data
        is_rgb = self._im_layer_widget.value.rgb

        if self._im_layer_widget.value.ndim == 2:
            image = image[None, ...]

        if not is_rgb:
            image = color.gray2rgb(image)

        elif image.shape[-1] == 4:
            image = color.rgba2rgb(image)

        if np.issubdtype(image.dtype, np.floating):
            image = image - image.min()
            image = image / image.max()

        image = util.img_as_float(image)
        init_frame = image[-1].astype(np.float32)

        self._mask_layer.data = np.zeros(
            (1, *init_frame.shape[-3:-1]), dtype=int
        )
        self._labels_layer.data = np.zeros(
            (1, *init_frame.shape[-3:-1]), dtype=int
        )

        self._image_predictor.set_image(init_frame)

        s = self._predictor.image_size
        image = transform.resize(
            image, (image.shape[0], s, s), order=1, anti_aliasing=True
        )
        image = np.transpose(image, (0, 3, 1, 2))  # TYXC -> TCYX

        tensor = torch.tensor(image, dtype=torch.float32, device=self._device)

        tensor -= img_mean
        tensor /= img_std

        tensor = torch.flip(tensor, (0,))

        return init_frame, tensor

    def _on_confirm_mask(self, _: Optional[Any] = None) -> None:
        if self._video is None:
            return

        labels = self._labels_layer.data
        mask = self._mask_layer.data
        labels[np.nonzero(mask)] = labels.max() + 1
        self._labels_layer.data = labels
        self._cancel_annot()

    def _cancel_annot(self, _: Optional[Any] = None) -> None:
        # boxes must be reset first because of how of points data update signal
        self._boxes_layer.data = []
        self._pts_layer.data = []
        self._mask_layer.data = np.zeros_like(self._mask_layer.data)

        self._confirm_mask_btn.enabled = False
        self._cancel_annot_btn.enabled = False
