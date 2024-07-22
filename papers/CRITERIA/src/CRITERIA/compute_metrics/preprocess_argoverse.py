# MIT License
#
# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import warnings
from pathlib import Path

import compress_pickle as comp_pkl
import cv2
import numpy as np
from argoverse.data_loading.argoverse_forecasting_loader import (
    ArgoverseForecastingLoader,
)
from argoverse.map_representation.map_api import ArgoverseMap
from scipy.ndimage.morphology import distance_transform_edt

from ..utils.common import get_argparse_parser, get_configuration

warnings.filterwarnings("ignore")  # ignore pandas copy warning.


def main(config: dict):
    prog_config = config[Path(__file__).stem]
    vis_map_path = prog_config.get("vis_map_path") or config["vis_map_path"]
    output_root = Path(vis_map_path).resolve()
    am = ArgoverseMap()
    for city_name in prog_config["city_names"]:
        print("Generating maps for {:s}.".format(city_name))

        mask_path = Path(output_root).joinpath(
            "raw_map", "{:s}_mask.pkl".format(city_name)
        )
        dt_path = Path(output_root).joinpath("raw_map", "{:s}_dt.pkl".format(city_name))
        mask_vis_path = Path(output_root).joinpath(
            "raw_map_visualization", "{:s}_mask_vis.png".format(city_name)
        )
        dt_vis_path = Path(output_root).joinpath(
            "raw_map_visualization", "{:s}_dt_vis.png".format(city_name)
        )
        mask_vis_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        map_mask, image_to_city = am.get_rasterized_driveable_area(city_name)

        print("Calculating Signed Distance Transform... ", end="", flush=True)
        image = map_mask.astype(np.int32)
        invert_image = 1 - image
        dt = np.where(
            invert_image,
            -distance_transform_edt(invert_image),
            distance_transform_edt(image),
        )
        print("Done.")

        print("Saving Results... ", end="", flush=True)
        comp_pkl.dump({"map": map_mask, "image_to_city": image_to_city}, mask_path)
        comp_pkl.dump({"map": dt, "image_to_city": image_to_city}, dt_path)

        mask_vis = (map_mask * 255).astype(np.uint8)

        dt_max = dt.max()
        dt_min = dt.min()
        dt_vis = ((dt - dt_min) / (dt_max - dt_min) * 255).astype(np.uint8)

        cv2.imwrite(str(mask_vis_path), mask_vis)
        cv2.imwrite(str(dt_vis_path), dt_vis)
        print(
            "Done. Saved {:s}, {:s}, {:s}, and {:s}.".format(
                str(mask_path), str(mask_vis_path), str(dt_path), str(dt_vis_path)
            )
        )


if __name__ == "__main__":
    parser = get_argparse_parser(Path(__file__).name)
    args = parser.parse_args()
    config = get_configuration(args)

    main(config)
