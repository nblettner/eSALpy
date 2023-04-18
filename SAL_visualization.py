import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
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
        
        
def colormap(
    cLevels=[0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    extend=2,
    belowColor=None,
    aboveColor=None,
    baseCmp=mpl.cm.YlGnBu,
    kind="listed",
):
    """
    Returns a colormap that is based on a standard map, with optional extra colors at beginning and end.

    extend: If colors are needed for values below and above range defined by cLevels, extend should be 2;
            if colors are needed for either below or above the range then extend=1;
            if only values within range are considered then extend=0.
            Extend does not make belowColor or aboveColor mandatory:
            if those are None, basemap colors are used so that they extend the range.

    belowColor and aboveColor: Give colors as hex value
    """

    # if specific edge colors are given, less base map colors are needed
    for edgeC in [belowColor, aboveColor]:
        if edgeC is not None:
            extend -= 1

    # number of colors that need to be filled by baseCmp.
    nBaseColors = len(cLevels) - 1 + extend

    # list of hex color codes of baseCmp.
    precipColors = []
    for p in np.linspace(0, 1, nBaseColors):
        rgb = baseCmp(p)[:3]
        precipColors.append(mpl.colors.rgb2hex(rgb))

    # add specific edge values
    if belowColor is not None:
        precipColors.insert(0, belowColor)
    if aboveColor is not None:
        precipColors.append(aboveColor)

    # transform to cmap
    if kind=="listed":
        cmap = ListedColormap(precipColors)
    elif kind=="segmented":
        cmap = LinearSegmentedColormap.from_list("mycmap", precipColors)

    return cmap
  


# def colormap(cLevels=None, values=None,
#              baseCmp=mpl.cm.YlGnBu(np.arange(256))[50:],
#              extend=2,
#              belowColor=None,
#              aboveColor=None,
#             ):
    
#     baseCmp = ListedColormap(baseCmp, name="myColorMap", N=baseCmp.shape[0])
    
#     if cLevels is None:
#         cLevels = np.linspace(np.nanmin(values),np.nanmax(values),30)
        
#     # if specific edge colors are given, less base map colors are needed
#     for edgeC in [belowColor, aboveColor]:
#         if edgeC is not None:
#             extend -= 1

#     # number of colors that need to be filled by baseCmp.
#     nBaseColors = len(cLevels) - 1 + extend

#     # list of hex color codes of baseCmp.
#     precipColors = []
#     for p in np.linspace(0, 1, nBaseColors):
#         rgb = baseCmp(p)[:3]
#         precipColors.append(mpl.colors.rgb2hex(rgb))

#     # add specific edge values
#     if belowColor is not None:
#         precipColors.insert(0, belowColor)
#     if aboveColor is not None:
#         precipColors.append(aboveColor)

#     # transform to cmap
#     cMap = ListedColormap(precipColors)

#     return cMap, cLevels


def visualization_SAL(field_sim, field_ref, sal_out, fig_width=5.5, cMap=None, cLevels=None, outname=None):

    # plot ____________
    font_small = 6
    font_title = 12
    c_markers = "#ef9400"
    cmap_binary_wb = ["#FFFFFF", "#000000"]
    cmap_binary_wo = ["#FFFFFF", c_markers]
    # plt.rcParams["font.size"] = 15  # 4
    fig_height = (1/1.4) * fig_width
    
    assert (field_sim.shape == field_ref.shape)
    
    y = np.arange(field_sim.shape[-2])
    x = np.arange(field_sim.shape[-1])

    plot_array = [
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
        [3, 3, 3, 3, 3, 3],
    ]

    fig, axs = pplt.subplots(
        plot_array,
        figsize=(fig_width, fig_height),
        space=0.1,
        sharey=3,
        sharex=3,
    )
    
    
    # ================================================
    # map plots
    
    title = ["Prediction", "Reference"]
    name = ["sim", "ref"]

    # loop over fields
    for i_field, field in enumerate([field_sim, field_ref]):

        # rainfall
        im = axs[i_field].pcolormesh(field, cmap=cMap, levels=cLevels, extend="both")

        # contour for threshold
        axs[i_field].contour(
            field,
            levels=np.array([-1, sal_out["thld_%s" % name[i_field]]]),
            colors=cmap_binary_wo,
        )

        # center of mass
        axs[i_field].scatter(
            sal_out["tcm_x_%s" % name[i_field]],
            sal_out["tcm_y_%s" % name[i_field]],
            marker="X",
            color=c_markers,
            ec="white",
            s=50,
            alpha=0.7,
            label=""
        )

        # format
        axs[i_field].set_title(title[i_field], fontsize=font_title)

    axs[:2].format(
        ylim=[y.min(), y.max()], xlim=[x.min(), x.max()], yticks=[], xticks=[],
        ylabel="", xlabel=""
    )

    axs[0].plot([], [], c_markers, label="Threshold Value")
    axs[0].scatter(
        [], [], marker="X", color=c_markers, ec="white", s=50, alpha=0.7, label="Center of Mass"
    )
    axs[0].legend(loc="lr", fontsize=font_small)
    cb = axs[1].colorbar(im, width=0.1, ticks=cLevels[::2], extend="both", extendsize=0.1)
    cb.ax.tick_params(labelsize=font_small) 
    
    
    # ================================================
    # SAL Plot 
    
    axs[2].scatter(
        sal_out["S"].values,
        3,
        c="r",
        marker="D",
        s=50,
        zorder=4
    )
    axs[2].scatter(
        sal_out["A"].values,
        2,
        c="r",
        marker="D",
        s=50,
        zorder=4
    )
    axs[2].scatter(
        sal_out["L"].values,
        1,
        c="r",
        marker="D",
        s=50,
        zorder=4
    )

    axs[2].axvline(0, c="k", linewidth=2, zorder=3)
    # axs[2].set_title('SAL', fontsize=14)

    # horizontal bars
    axs[2].axhline(3, c="gray")
    axs[2].axhline(2, c="gray")
    axs[2].axhline(1, xmin=0.5, c="gray")

    # formatting
    axs[2].set_ylim((0.5, 3.5))
    axs[2].set_yticks([1, 2, 3])
    axs[2].set_yticklabels(["L", "A", "S"], fontsize=font_title)
    axs[2].set_xlim((-2, 2))
    axs[2].set_xticks([-2, -1, 0, 1, 2])
    axs[2].set_xticklabels([-2, -1, 0, 1, 2], fontsize=font_title)
    axs[2].format(
        ygrid=False,
        ytickminor=False,
    )
    
    axs[2].yaxis.set_label_position("right")
    axs[2].yaxis.tick_right()
    
    if outname is not None:
        fig.savefig(outname, dpi=150)




def several_fields_in_row(fields_sim, field_ref, sal_list, fig_width=5.5, cMap=None, cLevels=None, outname=None):

    # plot ____________
    font_small = 4
    font_title = 8
    mrksz = 10
    cbwidth = 0.05
    c_markers = "#ef9400"
    cmap_binary_wb = ["#FFFFFF", "#000000"]
    cmap_binary_wo = ["#FFFFFF", c_markers]
    # plt.rcParams["font.size"] = 15  # 4
    fig_height = (1/1) * (4/3) * (1/len(fields_sim)) * fig_width
    
    for field_sim in fields_sim:
        assert (field_sim.shape == field_ref.shape)
    
    y = np.arange(field_sim.shape[-2])
    x = np.arange(field_sim.shape[-1])

    # define plot_array (layout)
    plot_arr_list = []
    for i in range(fields_sim.shape[0]+1):
        p = np.ones((4,3)) + (i*2)
        p[-1,:] = p[-1,:] + 1 
        plot_arr_list.append(p)
    plot_array = np.concatenate(plot_arr_list, axis=1)

    fig, axs = pplt.subplots(
        plot_array,
        figsize=(fig_width, fig_height),
        space=0.05,
        sharey=0,
        sharex=0,
    )
    
    
    # ================================================
    # map plots

    # ======================
    # ref field
    
    # rainfall
    im = axs[0].pcolormesh(field_ref, cmap=cMap, levels=cLevels, extend="both")

    # contour for threshold
    axs[0].contour(
        field_ref,
        levels=np.array([-1, sal_list[0]["thld_ref"]]),
        colors=cmap_binary_wo,
    )

    # center of mass
    axs[0].scatter(
        sal_list[0]["tcm_x_ref"],
        sal_list[0]["tcm_y_ref"],
        marker="X",
        color=c_markers,
        ec="white",
        s=mrksz,
        alpha=0.7,
        label=""
    )

    # format
    axs[0].set_title("Reference", fontsize=font_title)
    
    axs[0].format(
            ylim=[y.min(), y.max()],
            xlim=[x.min(), x.max()],
            yticks=[], xticks=[],
            ylabel="", xlabel="",
            aspect="equal"
        )
    
    axs[0].spines['bottom'].set_linewidth(2)
    axs[0].spines['top'].set_linewidth(2)
    axs[0].spines['left'].set_linewidth(2)
    axs[0].spines['right'].set_linewidth(2)
    
    axs[1].set_visible(False)
    
    # =======================
    # sim fields
    
    # loop over fields
    for i_field, field in enumerate(fields_sim):

        axi = 2*i_field + 2
        # rainfall
        im = axs[axi].pcolormesh(field, cmap=cMap, levels=cLevels, extend="both")

        # contour for threshold
        axs[axi].contour(
            field,
            levels=np.array([-1, sal_list[i_field]["thld_sim"]]),
            colors=cmap_binary_wo,
        )

        # center of mass
        axs[axi].scatter(
            sal_list[i_field]["tcm_x_sim"],
            sal_list[i_field]["tcm_y_sim"],
            marker="X",
            color=c_markers,
            ec="white",
            s=mrksz,
            alpha=0.7,
            label=""
        )

        # format
        axs[axi].set_title("Simulation: %i"%(i_field+1), fontsize=font_title)
        axs[axi].format(
            ylim=[y.min(), y.max()],
            xlim=[x.min(), x.max()],
            yticks=[], xticks=[],
            ylabel="", xlabel="",
            aspect="equal"
        )
        
        # ================================================
        # SAL Plot 

        axs[axi+1].scatter(
            sal_list[i_field]["S"].values,
            3,
            c="r",
            marker="D",
            s=mrksz,
            zorder=4
        )
        axs[axi+1].scatter(
            sal_list[i_field]["A"].values,
            2,
            c="r",
            marker="D",
            s=mrksz,
            zorder=4
        )
        axs[axi+1].scatter(
            sal_list[i_field]["L"].values,
            1,
            c="r",
            marker="D",
            s=mrksz,
            zorder=4
        )

        axs[axi+1].axvline(0, c="k", linewidth=2, zorder=3)
        # axs[2].set_title('SAL', fontsize=14)

        # horizontal bars
        axs[axi+1].axhline(3, c="gray")
        axs[axi+1].axhline(2, c="gray")
        axs[axi+1].axhline(1, xmin=0.5, c="gray")

        # formatting
        axs[axi+1].set_ylim((0.5, 3.5))
        axs[axi+1].set_xlim((-2, 2))
        axs[axi+1].set_xticks([-2, 0, 2])
        axs[axi+1].set_xticklabels([-2, 0, 2], fontsize=font_title)
        axs[axi+1].format(
            ygrid=False,
            ytickminor=False,
            yticks=[],
            xtickminor=False,
        )

    axs[-1].yaxis.set_label_position("right")
    axs[-1].yaxis.tick_right()
    axs[-1].set_yticks([1, 2, 3])
    axs[-1].set_yticklabels(["L", "A", "S"], fontsize=font_title)
    axs[-1].format(yticks=[1, 2, 3],
                   yticklabels=["L", "A", "S"])


    axs[0].plot([], [], c_markers, label="Threshold Value")
    axs[0].scatter(
        [], [], marker="X", color=c_markers, ec="white", s=mrksz, alpha=0.7, label="Center of Mass"
    )
    axs[0].legend(loc="lr", fontsize=font_small)
    cb = axs[-2].colorbar(im, width=cbwidth, ticks=cLevels[1::2], extend="both", extendsize=0.1, shrink=0.8)
    cb.ax.tick_params(labelsize=font_small) 
    
    
    if outname is not None:
        fig.savefig(outname, dpi=200)


