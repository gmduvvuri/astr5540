import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
sns.set_context('talk')


def test_func(x, y):
    return np.exp(-x * y)


def euler_step(func, x_old, y_old, step):
    return y_old + step * func(x_old, y_old)


def rk4_step(func, x_old, y_old, step):
    step2 = step / 2.0
    step6 = step / 6.0

    k1 = func(x_old, y_old)
    k2 = func(x_old + step2, y_old + step2 * k1)
    k3 = func(x_old + step2, y_old + step2 * k2)
    k4 = func(x_old + step, y_old + step * k3)

    return y_old + step6 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rational_extrapolate(y_arr, step):
    ny = len(y_arr)

    ans = np.empty((ny))
    y_old = np.zeros_like(y_arr)

    for m in range(1, ny):
        for s in range(0, ny - m):
            factor = 1.0 - (y_arr[s + 1] - y_arr[s]) / \
                (y_arr[s + 1] - y_old[s + 1])
            factor = factor * (step[s] / step[s + m])**2.0 - 1.0
            ans[s] = y_arr[s + 1] + (y_arr[s + 1] - y_arr[s]) / factor
        y_old = y_arr
        y_arr = ans
    return ans[0]


def bs_step(func, x_old, y_old, step, extrapolate_r=[]):
    if len(extrapolate_r) == 0:
        extrapolate_r = 5
    yex = np.empty((extrapolate_r))
    k = np.empty_like(yex)

    for j in range(0, extrapolate_r):
        nsub = 2.0 * float(j + 1.0)
        k[j] = step / nsub

        yn = y_old
        xn = x_old
        ynn = y_old + k[j] * func(x_old, y_old)
        xnn = x_old + k[j]

        for i in range(1, int(nsub), 2):
            yn = yn + 2.0 * k[j] * func(xnn, ynn)
            xn = xn + 2.0 * k[j]
            ynn = ynn + 2.0 * k[j] * func(xn, yn)
            xnn = xn + 2.0 * k[j]

        ynn = ynn - k[j] * func(xn, yn)
        yex[j] = (yn + ynn) / 2.0
    return rational_extrapolate(yex, k)


def drive(func, nsteps, extrapolate_r=[], nobs=[]):
    if len(nobs) == 0:
        nobs = 0

    dx = 1.0 / float(nsteps)
    x_old = 0.0
    y_old_euler = 0.0
    y_old_rk4 = 0.0
    y_old_bs = 0.0

    y_euler_array = np.zeros((nsteps))
    y_rk4_array = np.zeros_like(y_euler_array)
    y_bs_array = np.zeros_like(y_euler_array)
    x_array = np.zeros_like(y_euler_array)

    for i in range(0, nsteps):
        y_euler_array[i] = y_old_euler
        y_rk4_array[i] = y_old_rk4
        y_bs_array[i] = y_old_bs
        x_array[i] = x_old

        y_new_euler = euler_step(func, x_old, y_old_euler, dx)
        # y_new_rk4 = rk4_step(func, x_old, y_old_rk4, dx)
        if nobs == 0:
            pass
            # y_new_bs = bs_step(func, x_old, y_old_bs, dx)
        else:
            y_new_bs = 0.0
        x_old = x_old + dx
        y_old_euler = y_new_euler
        # y_old_rk4 = y_new_rk4
        # y_old_bs = y_new_bs

    y_euler_array[-1] = y_old_euler
    y_rk4_array[-1] = y_old_rk4
    y_bs_array[-1] = y_old_bs
    x_array[-1] = x_old
    true_y = 0.773877305

    euler_diff = np.abs(true_y - y_new_euler)
    rk4_diff = np.abs(true_y - y_new_rk4)
    bs_diff = np.abs(true_y - y_new_bs)

    print('Euler error is: ', euler_diff)
    print('4th order Runge-Kutta error is: ', rk4_diff)
    print('Burlisch-Stoer error is: ', bs_diff)

    return y_euler_array, y_rk4_array, y_bs_array, x_array


if __name__ == '__main__':
    '''
    for i in range(30, 33):
        euler, rk4, bs, x = drive(test_func, nsteps=int(2**i))
        titlename = 'Number of steps: ' + str(int(2**i))
        savename = 'step_num_' + str(int(2**i)) + '.pdf'


        true_y = 0.773877305

        euler_diff = np.abs(true_y - euler[-1])
        rk4_diff = np.abs(true_y - rk4[-1])
        bs_diff = np.abs(true_y - bs[-1])

        euler_label = 'Euler error = ' + "{:.3E}".format(euler_diff)
        rk4_label = '4th Order Runge-Kutta error = '
                    + "{:.3E}".format(rk4_diff)
        bs_label = 'Burlisch-Stoer error = ' + "{:.3E}".format(bs_diff)

        plt.figure(figsize=(10, 5))
        plt.plot(x, euler, ':', label=euler_label)
        plt.plot(x, rk4, '--', label=rk4_label, alpha=0.8)
        plt.plot(x, bs, '-', label=bs_label, alpha=0.4)
        plt.legend(loc='lower right')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title(titlename)
        plt.savefig(savename)
        plt.clf()
    '''
    x = 0.0
    y = 0.0
    dx = 1.0 / (2.0**28.0)
    true_y = 0.773877305
    while x < 1.0:
        print(x)
        y = euler_step(test_func, x, y, dx)
        x = x + dx
    euler_diff = np.abs(true_y - y)
    print('Euler error = ' + "{:.3E}".format(euler_diff))
