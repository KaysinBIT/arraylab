"""Plot planar fields into an existing Figure, or Agg PNGs without Qt."""
import numpy as np


def draw_planar_figure(figure, report, view, chinese=False):
    figure.clear()
    patterns = report["electrical"]["patterns"]
    p = report["geometry"]
    output = patterns["output"]
    if view == "cut":
        ax = figure.add_subplot(111)
        top=max(3,max(float(np.max(patterns[key]["cut_db"])) for key in patterns))
        for key, label in [("target", "Target"), ("uncompensated", "Uncompensated"), ("output", "Output")]:
            ax.plot(patterns[key]["cut_angles_deg"], np.maximum(patterns[key]["cut_db"], -60), label=label, lw=1.7)
        ax.set(xlim=(-90,90),ylim=(-60,top),xlabel="Signed theta (deg)",ylabel="Target coherent-sum reference (dB)",
               title=f"Planar cut: phi={p['cut_phi_deg']:g} deg")
        ax.grid(True,alpha=.3);ax.legend(fontsize=8)
    elif view == "map":
        ax = figure.add_subplot(111)
        theta = np.array(output["sphere_theta_deg"]);phi = output["sphere_phi_deg"]
        values = np.array(output["sphere_db"])[theta<=90]
        mesh = ax.pcolormesh(phi,theta[theta<=90],np.maximum(values,-40),vmin=-40,vmax=max(0,float(np.max(values))),cmap="viridis",shading="auto")
        ax.set(xlabel="Phi (deg)",ylabel="Theta from +z (deg)",title="Output: upper hemisphere")
        ax.invert_yaxis();figure.colorbar(mesh,ax=ax,label="dB / target coherent sum")
    elif view == "3d":
        ax = figure.add_subplot(111,projection="3d")
        tt,pp = np.meshgrid(np.deg2rad(output["sphere_theta_deg"]),np.deg2rad(output["sphere_phi_deg"]),indexing="ij")
        db = np.array(output["sphere_db"])
        radius = 10**(np.maximum(db,-300)/20)
        if report["electrical"]["metrics"]["active_ports"] == 0:radius[:]=0
        from matplotlib import colormaps
        colors = colormaps["viridis"](np.clip((db+40)/40,0,1))
        ax.plot_surface(radius*np.sin(tt)*np.cos(pp),radius*np.sin(tt)*np.sin(pp),radius*np.cos(tt),
                        facecolors=colors,rstride=1,cstride=2,linewidth=0,antialiased=False,shade=False)
        ax.set(xlabel="x",ylabel="y",zlabel="z",title="Full sphere: radius = relative field magnitude")
        bound=max(1,float(np.max(radius)))
        ax.set_xlim(-bound,bound);ax.set_ylim(-bound,bound);ax.set_zlim(-bound,bound)
        ax.set_box_aspect((1,1,1));ax.view_init(elev=22,azim=-50)
    else:
        ax = figure.add_subplot(111)
        ports = report["network"]["ports"]
        x,y = [r["x_lambda"] for r in ports],[r["y_lambda"] for r in ports]
        ax.scatter(x,y,c=[r["output_amplitude"] for r in ports],cmap="viridis",s=55,edgecolors="#24516a")
        if len(ports)<=36:
            for row in ports: ax.annotate(str(row["port"]),(row["x_lambda"],row["y_lambda"]),xytext=(4,5),textcoords="offset points",fontsize=8)
        ax.set(xlabel="x / wavelength",ylabel="y / wavelength",title="Planar element coordinates (z = 0)")
        ax.set_aspect("equal",adjustable="datalim");ax.grid(True,alpha=.3)


def render_planar_plots(directory, report, mode):
    import matplotlib
    # Explicit Agg canvas also works when called from the live Qt application;
    # do not change that application's global backend during export.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    kinds = ["cut","map","3d"] if mode=="all" else [mode]
    for kind in kinds:
        with matplotlib.rc_context({"font.family":"DejaVu Sans","font.size":9,"text.usetex":False}):
            figure = Figure(figsize=(8,6),dpi=140,layout="constrained");FigureCanvasAgg(figure)
            draw_planar_figure(figure,report,kind)
            figure.savefig(directory/f"planar-{kind}.png");figure.clear()
