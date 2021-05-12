# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import unittest

import torch

from ultra.baselines.common.social_vehicle_config import get_social_vehicle_configs


class EncoderTest(unittest.TestCase):
    def test_encoder_bias(self):
        ENCODER_NAMES = [
            "precog_encoder",
            "pointnet_encoder",
            "pointnet_encoder_batched",
        ]
        NUM_SOCIAL_FEATURES = 4
        NUM_SOCIAL_VEHICLES = 5

        for encoder_name in ENCODER_NAMES:
            social_vehicle_config_dict = {
                "encoder_key": encoder_name,
                "social_policy_hidden_units": 128,
                "social_policy_init_std": 0.5,
                "social_capacity": NUM_SOCIAL_VEHICLES,
                "num_social_features": NUM_SOCIAL_FEATURES,
                "seed": 2,
            }
            social_vehicle_config = get_social_vehicle_configs(
                **social_vehicle_config_dict
            )

            encoder_class = social_vehicle_config["encoder"][
                "social_feature_encoder_class"
            ]
            encoder_params = social_vehicle_config["encoder"][
                "social_feature_encoder_params"
            ]
            encoder_params["bias"] = True
            encoder_network_with_bias = encoder_class(**encoder_params)
            encoder_params["bias"] = False
            encoder_network_without_bias = encoder_class(**encoder_params)

            input_tensor = torch.zeros(
                size=(1, NUM_SOCIAL_VEHICLES, NUM_SOCIAL_FEATURES)
            )

            encoder_network_with_bias.eval()
            encoder_network_without_bias.eval()
            with torch.no_grad():
                with_bias_output_tensor = encoder_network_with_bias(input_tensor)
                with_bias_output_tensor = with_bias_output_tensor[0][0]
                without_bias_output_tensor = encoder_network_without_bias(input_tensor)
                without_bias_output_tensor = without_bias_output_tensor[0][0]

            # Expect the output tensor to be all zeros if there is no bias in the
            # network. The shape of this zero tensor can be like with_bias_output_tensor
            # or without_bias_output_tensor since the two output tensors should be the
            # same shape.
            without_bias_expected_tensor = torch.zeros_like(with_bias_output_tensor)

            self.assertTrue(
                not torch.equal(with_bias_output_tensor, without_bias_expected_tensor),
                msg=f"Failed on {encoder_name}.",
            )
            self.assertTrue(
                torch.equal(without_bias_output_tensor, without_bias_expected_tensor),
                msg=f"Failed on {encoder_name}.",
            )
