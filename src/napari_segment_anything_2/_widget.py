from typing import Optional, Tuple

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
from napari.layers import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
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

        self._run_btn = PushButton(text="Auto. Segm.")
        self._run_btn.changed.connect(self._on_run)
        self.append(self._run_btn)

        self._tracks_ckbox = CheckBox(text="Show tracks")
        self._tracks_ckbox.value = False
        self.append(self._tracks_ckbox)

        self._image: Optional[np.ndarray] = None
        self._model_type_widget.changed.emit(model_type)
        self._im_layer_widget.changed.emit(self._im_layer_widget.value)

    def _load_model(self, model_type: str) -> None:
        self._predictor = build_sam2_video_predictor(
            model_type, get_weights_path(model_type)
        )
        self._predictor.fill_hole_area = 0
        self._mask_generator = SAM2AutomaticMaskGenerator(self._predictor)

    def _load_image(self, im_layer: Optional[Image]) -> None:
        is_valid = False
        if im_layer is not None:
            image = im_layer.data
            if im_layer.ndim != 3 or im_layer.ndim != 2:
                print(
                    "Image must have 2 (YX) or 3 (TYX) plus optional RGB channels. "
                    f"Got {image.ndim}"
                )
            else:
                is_valid = True

        self._run_btn.enabled = is_valid

    @wait_cursor()
    def _on_run(self) -> None:
        if self._im_layer_widget.value is None:
            return

        init_frame, video = self._load_video()

        video = torch.flip(video, (0,))
        video_masks = np.zeros(
            (video.shape[0], video.shape[2], video.shape[3]), dtype=np.int32
        )

        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):

            state = self._predictor.init_state(video)
            self._predictor.reset_state(state)

            masks = self._mask_generator.generate(init_frame)
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

        self._viewer.add_labels(video_masks, name="Masks")

        if self._tracks_ckbox.value:
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

        s = self._predictor.image_size
        image = transform.resize(
            image, (image.shape[0], s, s), order=1, anti_aliasing=True
        )
        image = np.transpose(image, (0, 3, 1, 2))  # TYXC -> TCYX

        tensor = torch.tensor(image, dtype=torch.float32, device=self._device)

        tensor -= img_mean
        tensor /= img_std

        return init_frame, tensor
