from numpy import concatenate, ndarray
import pyuda


def mtanh_parameter_samples(
    shot: int, field_name: str, uda_client: pyuda.Client
) -> ndarray:

    assert field_name in ["t_e", "n_e"]
    group = f"/apf/core/mtanh/lfs/{field_name}/"
    mtanh_params = [
        "pedestal_location",
        "pedestal_height",
        "pedestal_width",
        "pedestal_top_gradient",
        "background_level",
    ]
    return concatenate(
        [
            uda_client.get(group + f"{p}_samples", shot).data[:, :, None]
            for p in mtanh_params
        ],
        axis=2,
    )


def exspline_parameter_samples(
    shot: int, field_name: str, uda_client: pyuda.Client
) -> ndarray:
    assert field_name in ["t_e", "n_e"]
    group = f"/apf/core/exspline/lfs/{field_name}/"

    # pull the sample data for all exspline parameters
    log_loc_samples = uda_client.get(group + "logistic_location_samples", shot).data
    log_floor_samples = uda_client.get(group + "logistic_floor_samples", shot).data
    log_width_samples = uda_client.get(group + "logistic_width_samples", shot).data
    basis_weight_samples = uda_client.get(group + "basis_weights_samples", shot).data

    # concatenate the parameters in the correct order to be passed to the exspline model
    return concatenate(
        [
            log_loc_samples[:, :, None],
            log_floor_samples[:, :, None],
            log_width_samples[:, :, None],
            basis_weight_samples,
        ],
        axis=2,
    )

