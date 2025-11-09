import matplotlib.pyplot as plt
from numpy import linspace
from inference.pdf import sample_hdi
from pyuda import Client
client = Client()

from apf_tools.profiles import exspline_profile_samples

# choose the radii at which to evaluate the profiles
radius = linspace(1.3, 1.5, 512)

# get the evaluated profiles for all the samples
profiles = exspline_profile_samples(
    shot=52448, radius=radius, uda_client=client, time_range=(0.5, 0.7), gradients=True
)

# get a particular time-slice
time = profiles["time"]
target_time = 0.613
time_index = abs(time - target_time).argmin()

# extract the profiles for the chosen time-slice
te = profiles["temperature_profiles"][time_index, :, :]
ne = profiles["density_profiles"][time_index, :, :]
te_grad = profiles["temperature_gradient_profiles"][time_index, :, :]
ne_grad = profiles["density_gradient_profiles"][time_index, :, :]

# combine the temperature / density profiles to get the pressure (in kPa)
pe = ne * te * 1.602e-19 * 1e-3

# get the pressure-gradient length-scale
pgls = (te_grad / te) + (ne_grad / ne)

# calculate the mean and highest-density interval for the PGLS
pgls_hdi = sample_hdi(pgls.T, fraction=0.95)
pgls_mean = pgls.mean(axis=1)

# plot all the calculated profiles
fig = plt.figure(figsize=(12, 7))
ax1, ax2, ax3, ax4 = [fig.add_subplot(2, 2, p) for p in range(1, 5)]
ax1.plot([0., 0.1], [0., 0.1], lw=2, c="red", label=r"$T_e$ samples")
ax1.plot(radius, te, lw=2, c="red", alpha=0.15)
ax1.plot([0., 0.1], [0., 0.1], lw=2, c="C0", label=r"$n_e$ samples")
ax1.plot(radius, ne * 1e-17, lw=2, c="C0", alpha=0.15)
ax1.set_xlim([1.35, 1.45])
ax1.set_xlabel("major radius (m)")
ax1.set_ylabel(r"$T_e$ (eV),    $n_e$ ($10^{17} \mathrm{m}^{-3}$)")
ax1.grid()
ax1.legend()

ax2.plot(radius, pe, lw=2, c="green", alpha=0.15)
ax2.set_xlim([1.35, 1.45])
ax2.set_xlabel("major radius (m)")
ax2.set_ylabel("electron pressure (kPa)")
ax2.grid()

ax3.plot(radius, pgls, lw=2, c="darkviolet", alpha=0.15)
ax3.set_xlim([1.35, 1.45])
ax3.set_xlabel("major radius (m)")
ax3.set_ylabel(r"pressure gradient length-scale $(\mathrm{m}^{-1})$" )
ax3.grid()

ax4.fill_between(radius, *pgls_hdi, color="darkviolet", alpha=0.3, label="95% HDI")
ax4.plot(radius, pgls_mean, color="darkviolet", lw=2, label="mean profile")
ax4.set_xlim([1.35, 1.45])
ax4.set_xlabel("major radius (m)")
ax4.set_ylabel(r"pressure gradient length-scale $(\mathrm{m}^{-1})$" )
ax4.grid()
ax4.legend()

plt.tight_layout()
plt.show()