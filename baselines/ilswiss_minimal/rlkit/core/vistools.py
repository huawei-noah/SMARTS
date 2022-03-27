import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

sns.set()


def plot_histogram(flat_array, num_bins, title, save_path):
    fig, ax = plt.subplots(1)
    ax.set_title(title)
    plt.hist(flat_array, bins=num_bins)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_2dhistogram(x, y, num_bins, title, save_path, ax_lims=None):
    fig, ax = plt.subplots(1)
    ax.set_title(title)
    plt.hist2d(x, y, bins=num_bins)
    if ax_lims is not None:
        ax.set_xlim(ax_lims[0])
        ax.set_ylim(ax_lims[1])
    ax.set_aspect("equal")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_seaborn_heatmap(x, y, num_bins, title, save_path, ax_lims=None):
    g = sns.kdeplot(x, y, cbar=True, cmap="RdBu")
    # g.set(title=title, xlim=tuple(ax_lims[0]), ylim=tuple(ax_lims[1]))
    g.set(xlim=tuple(ax_lims[0]), ylim=tuple(ax_lims[1]))
    g.figure.savefig(save_path)
    # g.figure.close()
    plt.close()


def plot_scatter(x, y, num_bins, title, save_path, ax_lims=None):
    fig, ax = plt.subplots(1)
    # ax.set_title(title)
    plt.scatter(x, y, s=0.5)
    if ax_lims is not None:
        ax.set_xlim(ax_lims[0])
        ax.set_ylim(ax_lims[1])
    ax.set_aspect("equal")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_seaborn_grid(grid, vmin, vmax, title, save_path):
    # ax = sns.heatmap(grid, vmin=vmin, vmax=vmax, cmap="YlGnBu")
    ax = sns.heatmap(grid, vmin=vmin, vmax=vmax, cmap="RdBu")
    # ax.set(title=title)
    ax.figure.savefig(save_path)
    plt.close()


def save_pytorch_tensor_as_img(tensor, save_path):
    if tensor.size(0) == 1:
        tensor = tensor.repeat(3, 1, 1)
    fig, ax = plt.subplots(1)
    ax.imshow(np.transpose(tensor.numpy(), (1, 2, 0)))
    plt.savefig(save_path)
    plt.close()


def generate_gif(list_of_img_list, names, save_path):
    fig, axarr = plt.subplots(len(list_of_img_list))

    def update(t):
        for j in range(len(list_of_img_list)):
            axarr[j].imshow(list_of_img_list[j][t])
            axarr[j].set_title(names[j])
        return axarr

    anim = FuncAnimation(
        fig, update, frames=np.arange(len(list_of_img_list[0])), interval=2000
    )
    anim.save(save_path, dpi=80, writer="imagemagick")
    plt.close()


def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    # for some weird reason 0 and 2 look almost identical
    n += 1
    cmap = plt.cm.get_cmap(name, n)
    # def new_cmap(n):
    #     if n >= 3: n = n+1
    #     if n == 1:
    #         return (0,0,0,1)
    #     else:
    #         return cmap(n)
    # return new_cmap
    return cmap


def plot_returns_on_same_plot(
    arr_list, names, title, save_path, x_axis_lims=None, y_axis_lims=None
):
    # print(arr_list, names, title, save_path, y_axis_lims)
    fig, ax = plt.subplots(1)
    cmap = get_cmap(len(arr_list))
    for i in range(len(arr_list)):
        cmap(i)

    for i, v in enumerate(zip(arr_list, names)):
        ret, name = v
        if ret.size <= 1:
            continue
        ax.plot(np.arange(ret.shape[0]), ret, color=cmap(i), label=name)

    ax.set_title(title)
    if x_axis_lims is not None:
        ax.set_xlim(x_axis_lims)
    if y_axis_lims is not None:
        ax.set_ylim(y_axis_lims)
    lgd = ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.05), shadow=False, ncol=3
    )
    plt.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.close()


def plot_multiple_plots(plot_list, names, title, save_path):
    fig, ax = plt.subplots(1)
    cmap = get_cmap(len(plot_list))

    for i, v in enumerate(zip(plot_list, names)):
        plot, name = v
        ax.plot(plot[0], plot[1], color=cmap(i), label=name)

    ax.set_title(title)
    lgd = ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.05), shadow=False, ncol=3
    )
    plt.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.close()


def save_plot(x, y, title, save_path, color="cyan", x_axis_lims=None, y_axis_lims=None):
    fig, ax = plt.subplots(1)
    ax.plot(x, y, color=color)
    ax.set_title(title)
    if x_axis_lims is not None:
        ax.set_xlim(x_axis_lims)
    if y_axis_lims is not None:
        ax.set_ylim(y_axis_lims)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_forward_reverse_KL_rews():
    plt.rcParams.update({"font.size": 16})
    # plt.rcParams.update({'lines.linewidth': 4})
    plot_line_width = 4
    line_color = "deepskyblue"

    # reverse KL
    fig, ax = plt.subplots(1)
    ax.plot(
        np.arange(-10, 10, 0.05),
        np.arange(-10, 10, 0.05),
        color=line_color,
        linewidth=plot_line_width,
    )
    ax.set_xlim([-10, 10])
    ax.set_ylim([-12, 12])
    ax.set_xlabel(r"log$\frac{\rho^{exp}(s,a)}{\rho^\pi(s,a)}$", fontsize="xx-large")
    ax.set_ylabel("$r(s,a)$", fontsize="xx-large")
    plt.axhline(0, color="grey")
    plt.axvline(0, color="grey")
    plt.savefig("plots/junk_vis/rev_KL_rew.png", bbox_inches="tight", dpi=150)
    plt.close()

    # GAIL
    fig, ax = plt.subplots(1)
    x = np.arange(-10, 10, 0.05)
    y = -np.log(1 + np.exp(-x))
    ax.plot(x, y, color=line_color, linewidth=plot_line_width)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-12, 12])
    ax.set_xlabel(r"log$\frac{\rho^{exp}(s,a)}{\rho^\pi(s,a)}$", fontsize="xx-large")
    ax.set_ylabel("$r(s,a)$", fontsize="xx-large")
    plt.axhline(0, color="grey")
    plt.axvline(0, color="grey")
    plt.savefig("plots/junk_vis/JS_rew.png", bbox_inches="tight", dpi=150)
    plt.close()

    # forward KL
    fig, ax = plt.subplots(1)
    x = np.arange(-10, 10, 0.05)
    y = np.exp(x) * (-x)
    ax.plot(x, y, color=line_color, linewidth=plot_line_width)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-2, 0.5])
    ax.set_xlabel(r"log$\frac{\rho^{exp}(s,a)}{\rho^\pi(s,a)}$", fontsize="xx-large")
    ax.set_ylabel("$r(s,a)$", fontsize="xx-large")
    plt.axhline(0, color="grey")
    plt.axvline(0, color="grey")
    plt.savefig("plots/junk_vis/forw_KL_rew.png", bbox_inches="tight", dpi=150)
    plt.close()


def _sample_color_within_radius(center, radius):
    x = np.random.normal(size=2)
    x /= np.linalg.norm(x, axis=-1)
    r = radius
    u = np.random.uniform()
    sampled_color = r * (u**0.5) * x + center
    return np.clip(sampled_color, -1.0, 1.0)


def _sample_color_with_min_dist(color, min_dist):
    new_color = np.random.uniform(-1.0, 1.0, size=2)
    while np.linalg.norm(new_color - color, axis=-1) < min_dist:
        new_color = np.random.uniform(-1.0, 1.0, size=2)
    return new_color


def visualize_multi_ant_target_percentages(
    csv_array, num_targets, title="", save_path=""
):
    # almost rainbow :P
    colors = ["purple", "cyan", "blue", "green", "yellow", "orange", "pink", "red"]

    # gather the results
    all_perc = [csv_array["Target_%d_Perc" % i] for i in range(num_targets)]
    all_dist = np.array(
        [csv_array["Target_%d_Dist_Mean" % i] for i in range(num_targets)]
    )
    all_dist[all_dist == -1] = 0.0
    all_dist = np.sum(all_dist * all_perc, axis=0)

    for i in range(1, len(all_perc)):
        all_perc[i] = all_perc[i] + all_perc[i - 1]

    X = np.arange(all_dist.shape[0])

    # time to plot
    plt.subplot(2, 1, 1)
    alpha = 0.5
    plt.fill_between(X, all_perc[0], color=colors[0], alpha=alpha)
    for i in range(1, len(all_perc)):
        plt.fill_between(
            X, all_perc[i], y2=all_perc[i - 1], color=colors[i], alpha=alpha
        )
    # plt.xlabel('epoch')
    plt.ylabel("Perc. Each Target")
    plt.title(title)

    plt.subplot(2, 1, 2)
    plt.plot(X, all_dist, color="royalblue")
    plt.ylim((0.0, 1.5))
    plt.xlabel("epoch")
    plt.ylabel("Dist. to Closest Target")

    if save_path == "":
        plt.savefig(
            "plots/junk_vis/test_multi_ant_plot.png", bbox_inches="tight", dpi=150
        )
    else:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def visualize_multi_ant_target_percentages_v2(
    csv_array, num_targets, title="", save_path=""
):
    # almost rainbow :P
    colors = ["purple", "cyan", "blue", "green", "yellow", "orange", "pink", "red"]

    # gather the results
    all_perc = [csv_array["Target_%d_Perc" % i] for i in range(num_targets)]
    all_dist = np.array(
        [csv_array["Target_%d_Dist_Mean" % i] for i in range(num_targets)]
    )
    all_dist[all_dist == -1] = 0.0
    # all_dist = np.sum(all_dist * all_perc, axis=0)

    for i in range(1, len(all_perc)):
        all_perc[i] = all_perc[i] + all_perc[i - 1]

    X = np.arange(all_dist[0].shape[0])

    # time to plot
    plt.subplot(num_targets + 1, 1, 1)
    alpha = 0.5
    plt.fill_between(X, all_perc[0], color=colors[0], alpha=alpha)
    for i in range(1, len(all_perc)):
        plt.fill_between(
            X, all_perc[i], y2=all_perc[i - 1], color=colors[i], alpha=alpha
        )
    # plt.xlabel('epoch')
    plt.ylabel("Perc. Each Target")
    plt.title(title)

    for i in range(num_targets):
        plt.subplot(num_targets + 1, 1, i + 2)
        plt.plot(X, all_dist[i], color=colors[i])
        plt.ylim((0.0, 3.5))
        plt.xlabel("epoch")
        plt.ylabel("Dist. to Closest Target")

    if save_path == "":
        plt.savefig(
            "plots/junk_vis/test_multi_ant_plot.png", bbox_inches="tight", dpi=150
        )
    else:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_fetch_pedagogical_example():
    import shapely.geometry as sg
    import descartes

    print(np.random.get_state())

    M = sg.Polygon([(-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)])

    v = np.random.uniform(-1.0, 1.0, size=2)
    good_inter = None
    bad_union = None
    good_circles = []
    bad_circles = []
    for i in range(6):
        u = _sample_color_within_radius(v, 0.5)
        f = _sample_color_with_min_dist(v, 0.5)

        good_c = sg.Point(*u).buffer(0.5)
        good_c = good_c.intersection(M)
        good_circles.append(good_c)
        bad_c = sg.Point(*f).buffer(0.5)
        bad_c = bad_c.intersection(M)
        bad_circles.append(bad_c)

        if good_inter is None:
            good_inter = good_c
        else:
            good_inter = good_inter.intersection(good_c)

        if bad_union is None:
            bad_union = bad_c
        else:
            bad_union = bad_union.union(bad_c)

        fig, ax = plt.subplots(1)
        for g in good_circles:
            ax.add_patch(descartes.PolygonPatch(g, fc="green", ec="green", alpha=0.1))
        for b in bad_circles:
            ax.add_patch(descartes.PolygonPatch(b, fc="pink", ec="pink", alpha=0.3))
        # ax.add_patch(descartes.PolygonPatch(bad_union, fc='pink', ec='pink', alpha=0.3))
        ax.add_patch(
            descartes.PolygonPatch(
                good_inter.difference(bad_union), fc="green", ec="green", alpha=1.0
            )
        )

        plt.plot(
            [v[0]],
            [v[1]],
            marker="*",
            markeredgecolor="gold",
            markerfacecolor="gold",
            markersize=20.0,
        )
        # markersize

        ax.set_xlim([-1.0, 1.0])
        ax.set_ylim([-1.0, 1.0])
        ax.set_aspect("equal")
        plt.savefig("plots/junk_vis/fetch/img_%d.png" % i, bbox_inches="tight", dpi=150)
        plt.close()

        # ax.set_xlabel(r'log$\frac{\rho^{exp}(s,a)}{\rho^\pi(s,a)}$', fontsize='xx-large')
        # ax.set_ylabel('$r(s,a)$', fontsize='xx-large')
        # plt.axhline(0, color='grey')
        # plt.axvline(0, color='grey')

    # # create the circles with shapely
    # a = sg.Point(-.5,0).buffer(1.)
    # b = sg.Point(0.5,0).buffer(1.)

    # # compute the 3 parts
    # left = a.difference(b)
    # right = b.difference(a)
    # middle = a.intersection(b)

    # # use descartes to create the matplotlib patches
    # ax = plt.gca()
    # ax.add_patch(descartes.PolygonPatch(left, fc='b', ec='k', alpha=0.2))
    # ax.add_patch(descartes.PolygonPatch(right, fc='r', ec='k', alpha=0.2))
    # ax.add_patch(descartes.PolygonPatch(middle, fc='g', ec='k', alpha=0.2))

    # # control display
    # ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)

    # plt.show()


if __name__ == "__main__":
    plot_forward_reverse_KL_rews()

    # csv_full_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-airl-rew-search/multi_target_ant_airl_rew_search_2019_05_04_14_07_14_0009--s-0/progress.csv'
    # csv_full_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-fairl-rew-search/multi_target_ant_fairl_rew_search_2019_05_04_14_08_18_0009--s-0/progress.csv'
    # csv_full_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-fairl-rew-search/multi_target_ant_fairl_rew_search_2019_05_04_14_28_51_0010--s-0/progress.csv'

    # fairl 64
    # csv_full_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-fairl-rew-search/multi_target_ant_fairl_rew_search_2019_05_04_14_08_17_0005--s-0/progress.csv'
    # csv_full_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-fairl-rew-search/multi_target_ant_fairl_rew_search_2019_05_04_14_28_51_0011--s-0/progress.csv'

    # import os
    # import json
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-fairl-rew-search/'
    # save_path = 'plots/junk_vis/fairl_multi_plot'
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-airl-rew-search/'
    # save_path = 'plots/junk_vis/airl_multi_plot'
    # deterministic ---------
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-fairl-rew-search-deterministic/'
    # save_path = 'plots/junk_vis/fairl_det_multi_plot'
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-airl-rew-search-deterministic/'
    # save_path = 'plots/junk_vis/airl_det_multi_plot'

    # 32 each
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-airl-rew-search-32-det-demos-per-task/'
    # save_path = 'plots/junk_vis/multi_ant_airl_32_det_demos_plot'
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-fairl-rew-search-32-det-demos-per-task/'
    # save_path = 'plots/junk_vis/multi_ant_fairl_32_det_demos_plot'

    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-fairl-rew-search-32-det-demos-per-task-state-only-rerun'
    # save_path = 'plots/junk_vis/multi_ant_fairl_32_det_demos_plot_state_only'
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-airl-rew-search-32-det-demos-per-task-state-only-rerun'
    # save_path = 'plots/junk_vis/multi_ant_airl_32_det_demos_plot_state_only'

    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-fairl-rew-search-32-det-demos-per-task-grad-pen-search/'
    # save_path = 'plots/junk_vis/multi_ant_fairl_32_det_demos_grad_pen_search'

    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-fairl-rew-search-32-det-demos-per-task-even-lower-grad-pen-search/'
    # save_path = 'plots/junk_vis/multi_ant_fairl_32_det_demos_even_lower_grad_pen_search'
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-airl-rew-search-32-det-demos-per-task-even-lower-grad-pen-search/'
    # save_path = 'plots/junk_vis/multi_ant_airl_32_det_demos_even_lower_grad_pen_search'

    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-fairl-32-det-demos-per-task-low-grad-pen-and-high-rew-scale-hype-search-0'
    # save_path = 'plots/junk_vis/multi_ant_fairl_32_det_demos_log_grad_high_rew_hype_search_0'

    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-4-directions-fairl-32-det-demos-per-task-hype-search-0-rb-size-3200-correct-final'
    # save_path = 'plots/junk_vis/multi_ant_fairl_32_det_demos_4_dir_hype_search_0_rb_size_3200'

    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-4-directions-fairl-32-det-demos-per-task-hype-search-1-rb-size-3200-correct-final'
    # save_path = 'plots/junk_vis/multi_ant_fairl_32_det_demos_4_dir_hype_search_1_rb_size_3200'

    # 4 distance ----------------------------
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-4-directions-4-distance-fairl-32-det-demos-per-task-hype-search-1-rb-size-3200-correct-final'
    # save_path = 'plots/junk_vis/multi_ant_fairl_32_det_demos_4_dir_4_distance_hype_search_1_rb_size_3200'

    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-4-directions-4-distance-fairl-32-det-demos-per-task-hype-search-1-rb-size-3200-correct-final-disc-128-2-tanh'
    # save_path = 'plots/junk_vis/multi_ant_fairl_32_det_demos_4_dir_4_distance_hype_search_1_rb_size_3200_disc_128_2_tanh'

    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-4-directions-4-distance-fairl-32-det-demos-per-task-hype-search-1-rb-size-3200-correct-final-disc-512-3-tanh'
    # save_path = 'plots/junk_vis/multi_ant_fairl_32_det_demos_4_dir_4_distance_hype_search_1_rb_size_3200_disc_512_3_tanh'

    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-4-directions-4-distance-fairl-32-det-demos-per-task-hype-search-1-rb-size-3200-correct-final-disc-128-2-tanh'
    # save_path = 'plots/junk_vis/multi_ant_fairl_32_det_demos_4_dir_4_distance_hype_search_1_rb_size_3200_disc_128_2_tanh'

    # 4 distance rel pos -------------------
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-rel-pos-4-directions-4-distance-fairl-32-det-demos-per-task-hype-search-1-rb-size-3200-correct-final-disc-512-3-relu'
    # save_path = 'plots/junk_vis/multi_ant_fairl_32_det_demos_rel_pos_4_dir_4_distance_hype_search_1_rb_size_4800_disc_512_3_relu'

    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-rel-pos-4-directions-4-distance-fairl-32-det-demos-per-task-hype-search-1-rb-size-3200-correct-final-disc-512-3-relu-high-rew-search'
    # save_path = 'plots/junk_vis/multi_ant_fairl_32_det_demos_rel_pos_4_dir_4_distance_hype_search_1_rb_size_4800_disc_512_3_relu_high_rew_search'

    # path terminates with 0.5 of target ----------------
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-rel-pos-4-directions-4-distance-fairl-32-det-demos-per-task-hype-search-1-rb-size-3200-correct-final-disc-512-3-relu-path-terminates-within-0p5-of-target-correct/'
    # save_path = 'plots/junk_vis/multi_target_ant_rel_pos_4_directions_4_distance_fairl_32_det_demos_per_task_hype_search_1_rb_size_3200_correct_final_disc_512_3_relu_path_terminates_within_0p5_of_target_correct'

    # tiny models hype search ---------------------------
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-rel-pos-with-termination-small-models-fairl-correct-disc-only-sees-rel-pos'
    # save_path = 'plots/junk_vis/multi_ant_tiny_fairl_disc_only_sees_rel_pos'

    # os.makedirs(save_path, exist_ok=True)

    # for sub_name in os.listdir(exp_path):
    #     if os.path.isdir(os.path.join(exp_path, sub_name)):
    #         csv_full_path = os.path.join(exp_path, sub_name, 'progress.csv')
    #         progress_csv = np.genfromtxt(csv_full_path, skip_header=0, delimiter=',', names=True)

    #         with open(os.path.join(exp_path, sub_name, 'variant.json')) as f:
    #             variant = json.loads(f.read())
    #         rew = variant['policy_params']['reward_scale']
    #         gp = variant['algo_params']['grad_pen_weight']
    #         title = 'rew_%d_grad_pen_%.2f' % (rew, gp)
    #         # visualize_multi_ant_target_percentages(progress_csv, 8, title=title, save_path=os.path.join(save_path, sub_name+'.png'))
    #         # visualize_multi_ant_target_percentages(progress_csv, 4, title=title, save_path=os.path.join(save_path, sub_name+'.png'))
    #         visualize_multi_ant_target_percentages_v2(progress_csv, 4, title=title, save_path=os.path.join(save_path, sub_name+'.png'))

    #         # visualize_multi_ant_target_percentages(progress_csv, 8, title=sub_name, save_path=os.path.join(save_path, sub_name+'.png'))

    # np.random.set_state(
    #     ('MT19937', np.array([4195284069, 4260559355, 3974968103, 2502666189,  383612239,
    #    3250448375, 2614363864, 1861504634, 3286089244, 1374357124,
    #     329811920, 2883451527, 4131935022, 1737734277, 1069442571,
    #    2318543242, 3644252591, 3974530001, 3418971723, 3149866447,
    #    2606705202, 1410132356, 1972514033,  826007982, 2620543685,
    #    1632910943, 4140656858, 1403276066, 2948759513, 3507343121,
    #     107140387, 1670727912, 2592868153, 3213445099, 2355158222,
    #     583458927, 3248123835, 1502714010,  384366037,  702202246,
    #    2750885330, 2492638776, 2896032739, 2672655546,  503533442,
    #    3752465378, 2257693035,  958611312, 1932937432,  140340639,
    #    1742552200, 1761930658, 2908258667, 1567366148, 3936644819,
    #    2558430596, 1601126393, 4258124678,  701517102, 1464351130,
    #    3389085943, 1717431082,  764285972,  492078668, 1247752310,
    #    3466738859,  365389004, 1677274195, 2703808654, 2277217010,
    #    1685114810, 2151650018, 3548837764, 1597518351, 2301150914,
    #    2052166042,   82893534, 1669328889, 2493000445, 3900856436,
    #    3477615070, 1468432145, 3049689567, 2005393300, 1276566131,
    #    3443682186, 2173102536, 1214888644, 4169778195,  702215807,
    #    1508387112, 3973918228, 2957348300, 3944938143, 3507260893,
    #    3169546984,  864000699, 4202156424, 3325765241, 2044503288,
    #    1751444469, 1445114575, 1134874702, 1854729234,  630769383,
    #     794768711, 1917701778, 1112203920,  724708148,  389070561,
    #    4069748720, 1975118490, 3611738192, 3072687406, 3944077891,
    #    4215525724,  844702869, 4087887514, 3583753350,  311982411,
    #    1746313362, 2459958727,  773756614, 1865804345, 2978000619,
    #    1517849394, 1392793970, 4275461254, 2253047694, 4025979091,
    #    2678471497, 1358368393, 1612387799, 3168562750,  389921108,
    #     292704504, 3277286795, 3002370908, 3717471681, 1997378094,
    #    2387759330, 4192348480,  316362162, 2423735574, 2225816215,
    #    3704236377, 1379876157, 3834846163, 2257136215,   51611065,
    #    2506005228, 1181522902,  952855134,  621253835,  440228656,
    #    3529725146, 1661746069, 4101577816, 2765273251, 2540685911,
    #     862812469, 4165784981,   40829880, 3363290005, 3683344588,
    #    1713863592,  665183992, 1508392389, 1360421608,  312768154,
    #    4215078941, 2206394487,  852169488, 2116461885, 1051694873,
    #     434796305, 4168803719, 4088288797, 4253020747, 3505920611,
    #    2070644973,  954075702, 2936356545,  650132753, 2856762573,
    #    3517204002, 1439205168,  968137126, 1835171795, 2928941441,
    #    2324006273, 1132584735, 3997080941, 2775102102, 3518792914,
    #     728356920, 1119237173,  583169432, 2181539188,  244259585,
    #     601670219, 2893395728, 1955980984, 2082156586, 1556366998,
    #    1099086944, 2062811964,  649317754, 1091179442, 3647161360,
    #    3299469926,  718425718,  380237830,  648716981,   28526471,
    #    2105354985, 1351560457, 1036187570, 3539609024, 1655925200,
    #    3168018653, 4293066248, 2330326812, 3156806102, 3801604623,
    #    2049048004, 4280107477, 1611063209, 2287998260, 1675465283,
    #     784614503,  107239043,  692156844, 1472304882, 1834763767,
    #    2309615711, 1352624721, 3070192386,  696811085,  438803518,
    #    4059081127,  425406636,  303674686, 3848603294, 3648378567,
    #    3690209205, 3606194271, 2337175958, 2440514353, 3568977519,
    #    3107099529, 3142603701, 3501577514,  111173742, 4054960709,
    #     153655851, 3066738609,  226108965, 1331770931, 3535765647,
    #    3506428257, 2233908617,  588180633,  739085970, 3735025914,
    #     683841423, 3641695526, 1537396183, 1520581291,  323102793,
    #    2114745701, 3514215686,  977209864, 2336349633, 4154095339,
    #    1936019017, 3130655313, 1535275125, 3462010460,   60032934,
    #    4143608592,  470536936, 1383162144,  599970243, 3384768455,
    #     640132659, 4261448873, 2618034420,   65289153,  540640172,
    #    1333479376, 2848048290, 2475899452,  605828257, 1109855130,
    #    3049875102, 1749683619,  596780343, 2042649664, 3000788873,
    #    1425487067, 2775645068, 1875195219, 1393281091, 3321585906,
    #    2783374608,  524234345, 3850424726, 2662984041,  300781360,
    #    4071223768, 1330745227, 2245778613, 1636839928, 4187914139,
    #     494558937, 4203557188, 2342169034, 2273843512,  883240171,
    #     121729608, 3431799018, 2980010267, 3665327330, 3415345879,
    #     644457383,  456853812, 2613092017, 1357968174, 4263006173,
    #    1078628050,  159993407, 1672336221, 3150345546, 1531878254,
    #    2280010018, 2117276119, 4032572907, 1539010694, 1362284624,
    #    3232267711, 1201055435,  199238098, 2455297224, 3405599679,
    #    1683076186, 4015492919, 4192677872, 4024720395, 2114925043,
    #    1007417821, 3575185253,  990997630, 2665066718, 2645742676,
    #    2602324948, 2427667244, 3496048609,  684754772,   86930734,
    #    1343514757, 2721858253, 1059237628,  187235323, 1480826512,
    #     897414676,  905823792, 2624422588, 1999054425, 2853991721,
    #    2825927244, 2009435512, 3242809435, 1093989703, 4001485750,
    #    3838563992, 3704267182, 2229709903, 3714410954, 4266599208,
    #    3848516823, 1361169307, 2078212880, 3415388908, 3563957691,
    #      94224306, 3871040139, 1956300044, 3011099099, 4085471881,
    #    2297844468, 2113952288, 1963825120, 2619904271, 3506796899,
    #    3371862084, 2187497141, 4231962461,   88347376, 4163133387,
    #     473958827, 1597863090, 1435879590, 1029059045, 1310414072,
    #    1118708536, 3493666300, 3698938803,  496729291,  997738542,
    #     878290479, 4035296902, 1630949308, 2636822193,  846271433,
    #    2941270010, 2244647613, 1594451291, 3462549181, 1046389067,
    #    4263743465, 2149108152, 3547954843,  487463993, 3807710597,
    #    3517124516, 1962414834,  258963350, 1363776890,  660147762,
    #     161101002, 2891777819,   26270302, 1081261975,  838804997,
    #    4059014828, 1676439313, 1274164273,   67410974, 2516662762,
    #    2285968911,  845705491, 2179390651, 3175198969, 1804892927,
    #    1696220658, 3936705871,  298668277, 1465671517, 2495392532,
    #    1207112213, 2280069405, 2664837184,  977996467,  314332792,
    #    3817259116,  612979885,  229377693, 1769865581, 4008250829,
    #     251078289,  403001258, 2597217779, 1357839918, 2815265832,
    #    3317734118, 1856524381,  196541831,  928740781, 1348988258,
    #     939893094, 1906777363, 3291673450, 2670033653,  836270807,
    #    1737108390, 2454357065, 1560226140, 2826602966, 2924145977,
    #    2310311266, 1360217496, 1761896571,  481973750,  943587119,
    #    1727707152,  615956948, 3451657153, 2403280550,  845473122,
    #    1965684676, 1194713663,  246009474, 3606564630,   66088733,
    #    3728644820,  202982220, 2287524363, 2832290832, 1038944091,
    #    1117998822, 1636140785,  975164602,  970772438, 3317273364,
    #    4264797890, 3741277628, 2684917256, 2578793635, 1841512512,
    #    2304022862,  527955907, 1992001728, 3207015409, 2103222735,
    #    2077999599,  671641361,  907451013, 4023352409,   54271515,
    #      75470221, 3766322146, 3688126658, 3159971077, 4292335330,
    #    3535822475, 3261254026, 1422101936, 1031048525, 3271467366,
    #    1802594666,  999101048, 1020695817,   86421679, 2715944314,
    #      93219897,  410549770, 4258710897, 2519238795, 3664457071,
    #    3925556711, 2193799409, 2499082230, 2374398319, 1003487773,
    #     323768095,    3940868, 1684673237, 3131102847, 1658837099,
    #     672573111, 2453653188, 2890165828, 1963089999,  992411316,
    #    1326162792,   73316872, 3878498976, 2363459019, 4194631049,
    #    3674195866, 1270475227, 1511339069, 3324289297, 1418562985,
    #    1579162995, 3946062686, 3318966627,  144590758,  304622518,
    #    1538673561,  931478509, 1000752613, 1372664611, 3195528484,
    #    3426210417, 2985090716,  807223848, 4099830420, 3116601741,
    #    1415637323, 2378399230, 2915129420, 4217945777,  814167187,
    #    4133205568, 3704708280,  666698340, 2857793714, 3553568012,
    #    1514657828, 1054094267, 2007934905, 2792452276, 3728319565,
    #    3825886808, 2030736147,  766267323, 3759752364, 3826898644,
    #    1582027499,  519156665,  743740361, 3202044104, 3069947825,
    #    3946066889, 2105725514,  759438248, 2308367044, 4015327801,
    #    3457164846, 3368296264,  309941066, 1151562548, 3540156972,
    #    1212358608,   41828404, 3820709256, 3060331732,  625972984,
    #    3157096328, 3246832567, 4174520076, 1890879156], dtype=np.uint32), 624, 0, 0.0)
    # )
    # plot_fetch_pedagogical_example()
