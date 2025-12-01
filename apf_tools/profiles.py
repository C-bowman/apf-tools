from numpy import ndarray, zeros
from pedinf.models import mtanh, exspline, ProfileModel
from apf_tools.parameters import exspline_parameter_samples, mtanh_parameter_samples
import pyuda


def exspline_profile_samples(
    shot: int,
    radius: ndarray,
    uda_client: pyuda.Client,
    gradients: bool = False,
    time_range: tuple[float, float] = None,
) -> dict[str, ndarray]:
    assert radius.ndim == 1
    time = uda_client.get(f"/apf/core/exspline/lfs/time", shot).data
    te_samples = exspline_parameter_samples(
        shot=shot, field_name="t_e", uda_client=uda_client
    )
    ne_samples = exspline_parameter_samples(
        shot=shot, field_name="n_e", uda_client=uda_client
    )

    if time_range is not None:
        assert len(time_range) == 2 and time_range[0] < time_range[1]
        t_inds = time_filtering(time, time_range)
        te_samples = te_samples[t_inds, :, :]
        ne_samples = ne_samples[t_inds, :, :]
        time = time[t_inds]

    basis_radius = uda_client.get(
        f"/apf/core/exspline/lfs/basis_function_radius", shot
    ).data

    model = exspline(knots=basis_radius, radius=radius)
    results = build_profiles(
        model=model, te_samples=te_samples, ne_samples=ne_samples, gradients=gradients
    )
    results["time"] = time
    return results


def mtanh_profile_samples(
    shot: int,
    radius: ndarray,
    uda_client: pyuda.Client,
    gradients: bool = False,
    time_range: tuple[float, float] = None,
) -> dict[str, ndarray]:
    assert radius.ndim == 1
    time = uda_client.get(f"/apf/core/mtanh/lfs/time", shot).data
    te_samples = mtanh_parameter_samples(
        shot=shot, field_name="t_e", uda_client=uda_client
    )
    ne_samples = mtanh_parameter_samples(
        shot=shot, field_name="n_e", uda_client=uda_client
    )

    if time_range is not None:
        assert len(time_range) == 2 and time_range[0] < time_range[1]
        t_inds = time_filtering(time, time_range)
        te_samples = te_samples[t_inds, :, :]
        ne_samples = ne_samples[t_inds, :, :]
        time = time[t_inds]

    model = mtanh(radius=radius)
    results = build_profiles(
        model=model, te_samples=te_samples, ne_samples=ne_samples, gradients=gradients
    )
    results["time"] = time
    return results


def build_profiles(
        model: ProfileModel, te_samples: ndarray, ne_samples: ndarray, gradients=False
):
    n_times, n_samples, _ = te_samples.shape
    n_radii = model.radius.size
    te = zeros([n_times, n_radii, n_samples])
    ne = zeros([n_times, n_radii, n_samples])

    for t in range(n_times):
        for s in range(n_samples):
            te[t, :, s] = model.forward_prediction(te_samples[t, s, :])
            ne[t, :, s] = model.forward_prediction(ne_samples[t, s, :])

    results = {"temperature_profiles": te, "density_profiles": ne}

    if gradients:
        te_grad = zeros([n_times, n_radii, n_samples])
        ne_grad = zeros([n_times, n_radii, n_samples])

        for t in range(n_times):
            for s in range(n_samples):
                te_grad[t, :, s] = model.forward_gradient(te_samples[t, s, :])
                ne_grad[t, :, s] = model.forward_gradient(ne_samples[t, s, :])

        results["temperature_gradient_profiles"] = te_grad
        results["density_gradient_profiles"] = ne_grad

    return results


def time_filtering(time: ndarray, time_range: tuple) -> ndarray:
    t_min, t_max = time_range
    assert t_min < t_max

    in_window = (time >= t_min) & (time <= t_max)
    if not in_window.any():
        raise ValueError("No data available inside the given time-window")

    return in_window.nonzero()[0]
