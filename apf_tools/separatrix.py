from numpy import linspace, ndarray
from scipy.interpolate import InterpolatedUnivariateSpline
from apf_tools.parameters import mtanh_parameter_samples
import pyuda


class MtanhElbowSpline:
    def __init__(self):
        z_position_data =  [
            -2.8610728, -2.7986620, -2.7217835, -2.6251872, -2.5011750, -2.3385879,
            -2.1225690, -1.8393692, -1.4959593, -1.1464081, -0.8607274, -0.6591142,
            -0.5225203, -0.4283038, -0.3609469, -0.3109975, -0.2727355, -0.2426087,
        ]

        self.spline = InterpolatedUnivariateSpline(
            x=linspace(-1.0, 1.0, len(z_position_data)), y=z_position_data
        )

    def __call__(self, theta: ndarray) -> ndarray:
        shape_factor = self.mtanh_shape_parameter(theta)
        z_positions = self.spline(shape_factor)
        return self.z_to_radius(z_positions, theta)

    @staticmethod
    def mtanh_shape_parameter(mtanh_samples: ndarray):
        R0, h, w, alpha, bg = [mtanh_samples[:, :, i] for i in range(5)]
        return (0.25 * alpha * w / (h - bg)).clip(-1.0, 1.0)

    @staticmethod
    def z_to_radius(z, theta):
        R0, h, w, alpha, bg = theta
        return R0 - 0.25 * z * w


def mtanh_elbow_position(
    shot: int, field_name: str, uda_client: pyuda.Client
) -> tuple[ndarray, ndarray]:
    samples = mtanh_parameter_samples(
        shot=shot, field_name=field_name, uda_client=uda_client
    )

    eblow_spline = MtanhElbowSpline()
    positions = eblow_spline(samples)
    return positions.mean(axis=1), positions.std(axis=1)
