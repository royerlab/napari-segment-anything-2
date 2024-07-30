# napari-segment-anything-2

[![License BSD-3](https://img.shields.io/pypi/l/napari-segment-anything-2.svg?color=green)](https://github.com/JoOkuma/napari-segment-anything-2/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-segment-anything-2.svg?color=green)](https://pypi.org/project/napari-segment-anything-2)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-segment-anything-2.svg?color=green)](https://python.org)
[![tests](https://github.com/JoOkuma/napari-segment-anything-2/workflows/tests/badge.svg)](https://github.com/JoOkuma/napari-segment-anything-2/actions)
[![codecov](https://codecov.io/gh/JoOkuma/napari-segment-anything-2/branch/main/graph/badge.svg)](https://codecov.io/gh/JoOkuma/napari-segment-anything-2)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-segment-anything-2)](https://napari-hub.org/plugins/napari-segment-anything-2)

A napari plugin for Meta's Segment Anything 2 in Images and Videos

https://github.com/user-attachments/assets/7ecc0e99-d6fd-42ad-bb2e-e903c04b6d9d

It works a bit better in natural images

https://github.com/user-attachments/assets/72037ceb-eabf-4222-bd2d-6c269ac582d6


----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-segment-anything-2` via [pip]:

    pip install git+https://github.com/jookuma/segment-anything-2@no-cc
    pip install napari[all] napari-segment-anything-2


To install latest development version :

    pip install git+https://github.com/jookuma/segment-anything-2@no-cc
    pip install napari[all] git+https://github.com/JoOkuma/napari-segment-anything-2.git


## Notes

To load mp4 or other video files, you need to install `napari_video`.

    pip install napari_video


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-segment-anything-2" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/JoOkuma/napari-segment-anything-2/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
