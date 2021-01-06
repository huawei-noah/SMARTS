from ultra.baselines.common.social_vehicle_extraction import *
from ultra.baselines.common.social_vehicles_encoders.pointnet_encoder import PNEncoder
from ultra.baselines.common.social_vehicles_encoders.pointnet_encoder_batched import (
    PNEncoderBatched,
)
from ultra.baselines.common.state_preprocessor import (
    StatePreprocessor,
    preprocess_state,
)
from ultra.baselines.common.social_vehicle_extraction import *
from ultra.baselines.common.social_vehicles_encoders.precog_encoder import (
    PrecogFeatureExtractor,
)

social_vehicle_extractors = {
    "pointnet_encoder": extract_social_vehicle_state_pointnet,
    "pointnet_encoder_batched": extract_social_vehicle_state_pointnet,
    "precog_encoder": extract_social_vehicle_state_default,
    "no_encoder": extract_social_vehicle_state_default,
}


def get_social_vehicle_configs(
    encoder_key,
    num_social_features,
    social_capacity,
    seed,
    social_policy_hidden_units=0,
    social_polciy_init_std=0,
):
    config = {
        "num_social_features": int(num_social_features),
        "social_capacity": int(
            social_capacity
        ),  # number of social vehicles to consider
        "social_vehicle_extractor_func": social_vehicle_extractors[encoder_key],
        "encoder_key": encoder_key,
    }
    if encoder_key == "precog_encoder":
        config["encoder"] = {  # state_size: 18 + social_capacity*output_dim
            "use_leading_vehicles": None,
            "social_feature_encoder_class": PrecogFeatureExtractor,
            "social_feature_encoder_params": {
                "hidden_units": int(social_policy_hidden_units),
                "n_social_features": int(num_social_features),
                "embed_dim": 8,
                "social_capacity": int(social_capacity),
                "seed": int(seed),
            },
        }
    elif encoder_key == "pointnet_encoder":
        config["encoder"] = {
            "use_leading_vehicles": None,
            "social_feature_encoder_class": PNEncoder,
            "social_feature_encoder_params": {
                "input_dim": int(num_social_features),
                "nc": 8,
                "transform_loss_weight": 0.1,
            },
        }
    elif encoder_key == "pointnet_encoder_batched":
        config["encoder"] = {
            "use_leading_vehicles": None,
            "social_feature_encoder_class": PNEncoderBatched,
            "social_feature_encoder_params": {
                "input_dim": int(num_social_features),
                "nc": 8,
                "transform_loss_weight": 0.1,
            },
        }
    elif encoder_key == "no_encoder":
        config["encoder"] = {  # state_size: 18 + social_capacity*social_features
            "use_leading_vehicles": {
                "social_capacity": int(social_capacity),
                "max_dist_social_vehicle": 100,
                "num_social_vehicle_per_lane": 2,
            },
            "social_feature_encoder_class": None,  # No Encoder; Examples at the bottom of this page
            "social_feature_encoder_params": {},
        }

    return config
