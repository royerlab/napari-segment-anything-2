import urllib.request
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from qtpy.QtCore import Qt
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QApplication

BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"

SAM_WEIGHTS_URL = {
    "sam2_hiera_b+": f"{BASE_URL}sam2_hiera_base_plus.pt",
    "sam2_hiera_l": f"{BASE_URL}sam2_hiera_large.pt",
    "sam2_hiera_s": f"{BASE_URL}sam2_hiera_small.pt",
    "sam2_hiera_t": f"{BASE_URL}sam2_hiera_tiny.pt",
}


@contextmanager
def wait_cursor():
    try:
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        yield
    finally:
        QApplication.restoreOverrideCursor()


def _report_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    percent = downloaded * 100 / total_size
    downloaded_mb = downloaded / 1024 / 1024
    total_size_mb = total_size / 1024 / 1024
    print(
        f"Download progress: {percent:.1f}% ({downloaded_mb:.1f}/{total_size_mb:.1f} MB)",
        end="\r",
    )


@wait_cursor()
def get_weights_path(model_type: str) -> Optional[Path]:
    """Returns the path to the weight of a given model architecture."""
    weight_url = SAM_WEIGHTS_URL[model_type]

    cache_dir = Path.home() / ".cache/napari-segment-anything"
    cache_dir.mkdir(parents=True, exist_ok=True)

    weight_path = cache_dir / weight_url.split("/")[-1]

    # Download the weights if they don't exist
    if not weight_path.exists():
        print(f"Downloading {weight_url} to {weight_path} ...")
        try:
            urllib.request.urlretrieve(
                weight_url, weight_path, reporthook=_report_hook
            )
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            urllib.error.ContentTooShortError,
        ) as e:
            warnings.warn(f"Error downloading {weight_url}: {e}", stacklevel=1)
            return None
        else:
            print("\rDownload complete.                            ")

    return weight_path
