import numpy as np
import matplotlib.pyplot as plt
import proplot as pplt




def maps_with_bars(fields, sal_out, cMap=None, cLevels=None, outname=None):

    mapsz = 7
    x_strech = 1.4
    fontsz = 14
    c_markers = "r"
    cmap_binary_wb = ["#FFFFFF", "#000000"]
    cmap_binary_wo = ["#FFFFFF", c_markers]

    plot_array = [
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
        [3, 3, 3, 3, 3, 3],
    ]

    fig, axs = pplt.subplots(
        plot_array,
        figsize=(mapsz * x_strech, mapsz),
        space=0.1,
        sharey=3,
        sharex=3,
    )

    # map plots -----------------
    title = ["Prediction", "Reference"]
    name = ["sim", "ref"]

    cLevels = np.linspace(0, np.max(fields), 20)
    # loop over fields
    for i_field, field in enumerate(fields):

        # rainfall
        axs[i_field].pcolormesh(field, cmap=cMap, levels=cLevels)

        # # contour for threshold
        # axs[i_field].contour(
        #     field,
        #     levels=np.array([-1, sal_out["thld_%s" % name[i_field]]]),
        #     lw=50,
        #     colors=cmap_binary_wo,
        #     zorder=4
        # )
        
        # center of mass
        axs[i_field].scatter(
            sal_out["tcm_x_%s" % name[i_field]],
            sal_out["tcm_y_%s" % name[i_field]],
            marker="X",
            color=c_markers,
            ec="white",
            s=200,
            label=""
        )

        # format
        axs[i_field].set_title(title[i_field], fontsize=fontsz)

    axs[:2].format(
        ylim=[0, field.shape[0]], xlim=[0, field.shape[1]], yticks=[], xticks=[],
        ylabel="", xlabel=""
    )

    # axs[0].plot([], [], c_markers, label="Threshold Value")
    axs[0].scatter(
        [], [], marker="X", color=c_markers, ec="white", s=200, label="Center of Mass"
    )
    axs[0].legend(loc="lr")  # , fontsize=14)

    # SAL Plot -------------------
    axs[2].scatter(
        sal_out["S"].values,
        3,
        c="r",
        marker="D",
        s=100,
        zorder=4
    )
    axs[2].scatter(
        sal_out["A"].values,
        2,
        c="r",
        marker="D",
        s=100,
        zorder=4
    )
    axs[2].scatter(
        sal_out["L"].values,
        1,
        c="r",
        marker="D",
        s=100,
        zorder=4
    )

    axs[2].axvline(0, c="k", linewidth=5, zorder=3)
    # axs[2].set_title('SAL', fontsize=14)

    # horizontal bars
    axs[2].axhline(3, c="gray")
    axs[2].axhline(2, c="gray")
    axs[2].axhline(1, xmin=0.5, c="gray")

    # formatting
    axs[2].set_ylim((0.5, 3.5))
    axs[2].set_yticks([1, 2, 3])
    axs[2].set_yticklabels(["L", "A", "S"], fontsize=fontsz)
    axs[2].set_xlim((-2, 2))
    axs[2].set_xticks([-2, -1, 0, 1, 2])
    axs[2].set_xticklabels([-2, -1, 0, 1, 2], fontsize=fontsz)
    axs[2].format(
        ygrid=False,
        ytickminor=False,
    )
    
    if outname is not None:
        fig.savefig(outname, dpi=150)
        
        
def gauss2D(y, x, ymu, xmu, sig):
    yp = (1 / (sig * (2 * np.pi) ** (1 / 2))) * np.exp(
        (-1 / 2) * ((y - ymu) / sig) ** 2
    )
    xp = (1 / (sig * (2 * np.pi) ** (1 / 2))) * np.exp(
        (-1 / 2) * ((x - xmu) / sig) ** 2
    )
    return yp * xp