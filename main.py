# Optimal Estimation - HW2 - Wind Simulator

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

# Matplotlib global variables
mpl.rcParams['legend.loc'] = 'lower right'
mpl.rcParams['lines.linewidth'] = 0.66
mpl.rcParams['lines.linestyle'] = '--'

def build_gust_mats(sigma_mat: np.array, L_mat: np.array):

    sigma_u = sigma_mat[0]
    sigma_v = sigma_mat[1]
    sigma_w = sigma_mat[2]
    L_u = L_mat[0]
    L_v = L_mat[1]
    L_w = L_mat[2]
    sqrtt = np.sqrt(3)

    A = np.array([[- v_inf / L_u, 0, 0, 0, 0], 
                  [0, - v_inf / L_v, sigma_v * (1 - sqrtt) * (v_inf / L_v)**(3 / 2), 0, 0], 
                  [0, 0, - v_inf / L_v, 0, 0], 
                  [0, 0, 0, - v_inf / L_w, sigma_w * (1 - sqrtt) * (v_inf / L_w)**(3 / 2)], 
                  [0, 0, 0, 0, - v_inf / L_w]])
    
    B = np.array([[sigma_u * (2 * v_inf / np.pi / L_u)**(1 / 2)], 
                  [sigma_v * (3 * v_inf / L_v)**(1 / 2)], 
                  [1], 
                  [sigma_w * (3 * v_inf /  L_w)**(1 / 2)], 
                  [1]])

    return A, B

def calc_time_hist(sigma_mat: np.array, L_mat: np.array):

    vel_old = np.zeros((5, 1))
    vel_hist = vel_old.T
    t_hist = [0]
    t = 0
    A, B = build_gust_mats(sigma_mat, L_mat)

    while t < t_total:

        dvel = A @ vel_old + B * sp.stats.norm.rvs(scale=np.sqrt(1/dt))
        # dvel = A @ vel_old + B * np.random.normal(0, np.sqrt(1/dt))
        vel_new = vel_old + dvel * dt
        vel_hist = np.vstack((vel_hist, vel_new.T))
        vel_old = vel_new
        t += dt
        t_hist.append(t)
    
    return vel_hist, t_hist

def main():

    global v_inf
    global alt
    global dt
    global t_total
    v_inf = 824     # ft/s
    alt = 20_000    # ft
    dt = 0.01       # s
    t_total = 600   # s

    sigma_mat_dict = {'light': np.array([5, 5, 5]),         # ft/sec
                      'medium': np.array([10, 10, 10]),     # ft/sec
                      'heavy': np.array([20, 20, 20]),      # ft/sec
                      }
    L_mat = np.array([1750, 1750, 1750])

    vel_hist_light, t_hist_light = calc_time_hist(sigma_mat_dict['light'], L_mat)
    vel_hist_medium, t_hist_medium = calc_time_hist(sigma_mat_dict['medium'], L_mat)
    vel_hist_heavy, t_hist_heavy = calc_time_hist(sigma_mat_dict['heavy'], L_mat)

    fig, ax = plt.subplots(3, 1)
    fig.suptitle(r'Gust Velocity ($u_g, v_g, w_g$) vs Time')
    fig.supxlabel(r'Time, $s$')
    fig.supylabel(r'Velocity, $ft/s$')
    ax[0].set_title(r'$u_g$ vs time')
    ax[0].plot(t_hist_light, vel_hist_light[:, 0], label=r'light, $\sigma_u=5$')
    ax[0].plot(t_hist_medium, vel_hist_medium[:, 0], label=r'medium, $\sigma_u=10$')
    ax[0].plot(t_hist_heavy, vel_hist_heavy[:, 0], label=r'heavy, $\sigma_u=20$')
    ax[0].legend()

    ax[1].set_title(r'$v_g$ vs time')
    ax[1].plot(t_hist_light, vel_hist_light[:, 1], label=r'light, $\sigma_v=5$')
    ax[1].plot(t_hist_medium, vel_hist_medium[:, 1], label=r'medium, $\sigma_v=10$')
    ax[1].plot(t_hist_heavy, vel_hist_heavy[:, 1], label=r'heavy, $\sigma_v=20$')
    ax[1].legend()

    ax[2].set_title(r'$w_g$ vs time')
    ax[2].plot(t_hist_light, vel_hist_light[:, 3], label=r'light, $\sigma_w=5$')
    ax[2].plot(t_hist_medium, vel_hist_medium[:, 3], label=r'medium, $\sigma_w=10$')
    ax[2].plot(t_hist_heavy, vel_hist_heavy[:, 3], label=r'heavy, $\sigma_w=20$')
    ax[2].legend()

    plt.show()

    return 1

if __name__=='__main__':
    main()