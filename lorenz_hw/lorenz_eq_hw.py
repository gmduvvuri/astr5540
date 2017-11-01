import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
sns.set_context('talk')


def lorenz_func(input, output, func_params):
    sigma, r, b = func_params
    x_slope = -sigma * (output[0] - output[1])
    y_slope = r * output[0] - output[1] - output[0] * output[2]
    z_slope = output[0] * output[1] - b * output[2]
    return np.array([x_slope, y_slope, z_slope])


def rk4_step(func, in_old, out_old, step, func_params):
    step2 = step / 2.0
    step6 = step / 6.0

    k1 = func(in_old, out_old, func_params)
    k2 = func(in_old + step2, out_old + step2 * k1, func_params)
    k3 = func(in_old + step2, out_old + step2 * k2, func_params)
    k4 = func(in_old + step, out_old + step * k3, func_params)

    return out_old + step6 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def get_txyz_lorenz(lorenz_params=[3.0, 17.0, 1.0],
                    init_conditions=[0.0, 1.0, 0.0],
                    step_size=1e-4, start_time=0.0, end_time=50.0):
    # lorenz_params = [sigma, r, b]

    t_array = np.arange(start_time, end_time + step_size, step_size)
    out_array = np.empty((3, len(t_array)))
    out_array[:, 0] = init_conditions_1

    for i in range(1, len(t_array)):
        print(t_array[i - 1], out_array[:, i - 1])
        out_array[:, i] = rk4_step(
            lorenz_func, t_array[i - 1],
            out_array[:, i - 1],
            step_size,
            lorenz_params_1)

    x_array = out_array[0, :]
    y_array = out_array[1, :]
    z_array = out_array[2, :]

    return t_array, x_array, y_array, z_array


def plot_phase_space(t, x, y, z, save_name='SHOW'):
    ax1, ax2 = plt.subplots((1, 2), sharey=True)

    ax1.plot(x, y)
    ax1.set_xlabel(r'$x$')
    ax1.set_yabel(r'$y$')
    ax1.set_title(r'$y$ vs $x$')

    ax2.plot(z, y)
    ax2.set_xlabel(r'$z$')
    ax2.set_title(r'$y$ vs $z$')

    plt.suptitle('Phase-Space Diagrams')

    if save_name == 'SHOW':
        plt.show()
    else:
        plt.savefig(save_name)
    plt.clf()


def plot_power_period(t, periodic_array, periodic_label=r'$x$',
                      upper_limit=None, save_name='SHOW'):
    ax1, ax2 = plt.subplots((2, 1))

    freqs = np.fft.ftt(len(t), t[1])
    power = (np.abs(np.fft.fft(periodic_array)))**2.0
    power_sort = power[np.argsort(freqs)]
    freq_sort = np.sort(freqs)
    if upper_limit:
        freq_mask = np.where(((freq_sort > 0.0) & (freq_sort < upper_limit)))
    else:
        freq_mask = np.where(freq_sort < 0.0)

    ax1.semilogy(freq_sort[freq_mask], power_sort[freq_mask])
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Power')

    ax2.plot(t, periodic_array)
    ax2.set_xlabel(r'$t$')
    ax2.set_ylabel(periodic_label)

    plt.suptitle('Periodicity Analysis')

    if save_name == 'SHOW':
        plt.show()
    else:
        plt.savefig(save_name)
    plt.clf()


def plot_poincare_x(t, x, y, z, save_name='SHOW'):
    zero_crossings = np.where(np.diff(np.signbit(x)))

    plt.scatter(z[zero_crossings], y[zero_crossings])
    plt.xlabel(r'$z$')
    plt.ylabel(r'$y$')
    plt.title(r'$\mathrm{Poincar\'e Map for }x=0$')

    if save_name == 'SHOW':
        plt.show()
    else:
        plt.savefig(save_name)
    plt.clf()
