import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


custom_colors = {
    'primary': '#e30513',
    'secondary': '#ffc103',
    'accent1': '#0099ff',
    'background': '#2a2a2a',
    'text': '#ffffff',
    'grid': '#ffffff',
    'line': '#ffffff'  # Line color in white for contrast
}
plt_param = {
    # 'font.family': 'Roboto',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.unicode_minus': False,
    'text.color': custom_colors['text'],
    'axes.labelcolor': custom_colors['text'],
    'xtick.color': custom_colors['text'],
    'ytick.color': custom_colors['text'],
    'axes.titlecolor': custom_colors['text'],
}
plt_style = 'dark_background'



def plot_spectra(wavelength, spectra_dict, mode="dark", title=None, legend=False):
    """
    Plots multiple absorption spectra with either a dark or white theme.

    Parameters:
    - wavelength (array-like): The wavelength array corresponding to the absorption values.
    - spectra_dict (dict): Dictionary containing spectra data. Keys are the labels for the legend.
    - mode (str): The theme mode. Accepts either 'dark' or 'white'.

    Returns:
    fig, ax: The figure and axes objects
    """
    # Setting up the figure

    if mode == "dark":
        bg_color = "dark_background"
        title_color = "white"
        label_color = "white"
        grid_color = "gray"
    elif mode == "white":
        bg_color = "default"  # plt.style.use('default')
        title_color = "black"
        label_color = "black"
        grid_color = "gray"
    else:
        raise ValueError("Invalid mode. Accepts either 'dark' or 'white'.")

    plt.style.use(bg_color)
    fig, ax = plt.subplots(figsize=(24, 12))

    # Loop through each spectrum in the dictionary and plot it
    for label, spectrum in spectra_dict.items():
        ax.plot(wavelength, spectrum, label=label, lw=4)

    # Customize plot aesthetics
    if title:
        ax.set_title(title, fontsize=24, fontweight="bold", color=title_color)
    else: 
        ax.set_title(label, fontsize=24, fontweight="bold", color=title_color)

    ax.set_xlabel("Wavelength (nm)", fontsize=24, fontweight="bold", color=label_color)
    ax.set_ylabel("reflectance", fontsize=24, fontweight="bold", color=label_color)
    ax.grid(True, linestyle="--", linewidth=0.5, color=grid_color)
    ax.tick_params(axis='both', which='major', labelsize=16, colors=label_color, width=2)

    if legend:
        leg = ax.legend(bbox_to_anchor=(.8, 1),
                        loc='upper left', borderaxespad=0.,
                        fontsize=24, # Increase font size
                        title_fontsize=36, # Set title font size
                        frameon=True, # Add a frame
                        fancybox=True, # Round corners
                        shadow=True, # Add shadow
                        ncol=1)
        # Make the legend background transparent
        leg.get_frame().set_alpha(0.1)

    plt.tight_layout()

    return fig, ax

# Usage:
# fig, ax = plot_spectra(radiometry.wvl, all_spectra, mode="dark", title="Reflectance Submodels", legend=True)
# plt.show()

def plot_multiple_dfs(df_dict, mode="dark"):
    """
    Plot multiple DataFrames on the same set of subplots.
    Parameters:
        df_dict (dict): Dictionary of Pandas DataFrames, where keys are the legends.
        mode (str): The background style of the plot. Either 'dark' or 'white'.
    Returns:
        None
    """
    # Set background style
    if mode == "dark":
        plt.style.use("dark_background")
    elif mode == "white":
        plt.style.use("default")
    else:
        print("Invalid mode. Using 'dark' as default.")
        plt.style.use("dark_background")

    # Pre-generate colors for each DataFrame
    color_dict = {
        legend: np.random.rand(
            3,
        )
        for legend in df_dict.keys()
    }

    # Create subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Flatten the axis array and iterate
    for ax, column in zip(axs.flatten(), list(df_dict.values())[0].columns):
        for legend, df in df_dict.items():
            ax.plot(
                df.index,
                df[column],
                marker="o",
                linestyle="-",
                linewidth=2,
                markersize=8,
                label=legend,
                color=color_dict[legend],
            )
        ax.set_title(column, fontsize=16, fontweight="bold")
        ax.set_xlabel("Index", fontsize=12, fontweight="bold")
        ax.set_ylabel("Value", fontsize=12, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.legend()
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Show plot
    plt.show()


def plot_multi_curve_refined_results(lines, results, mode="dark"):
    """
    Plots multiple curves and their refined maxima on the same plot.
    Parameters:
    - results (dict): Dictionary of results from multi_curve_refine function.
    - mode (str): Mode for the plot's appearance ('dark' or 'white').
    Returns:
    - A plot displaying multiple curves and their refined maxima.
    """
    if mode == "dark":
        bg_color = "dark_background"
        title_color = "white"
        label_color = "white"
        grid_color = "gray"
    elif mode == "white":
        bg_color = "default"  # plt.style.use('default')
        title_color = "black"
        label_color = "black"
        grid_color = "gray"
    else:
        raise ValueError("Invalid mode. Accepts either 'dark' or 'white'.")
    with plt.rc_context(
        {
            "axes.edgecolor": grid_color,
            "xtick.color": label_color,
            "ytick.color": label_color,
            "figure.facecolor": grid_color,
        }
    ):
        plt.figure(figsize=(12, 8))
        plt.style.use(bg_color)
        fig, ax = plt.subplots(figsize=(10, 8))
        for key, result in results.items():
            # if key in 'l_{ms}' :
            # Plot the refined curve
            (line,) = ax.plot(
                result["sorted_x_values"],
                result["sorted_y_values"] - np.nanmax(result["sorted_y_values"]),
                label=f"${key}$",
                linewidth=2,
            )
            current_color = line.get_color()
            ax.set_yscale("symlog")
            ax.set_ylim([-100, 0.5])
            # Highlight the refined local maxima
            refined_maxima_y_values = np.interp(
                result["refined_maxima"],
                result["sorted_x_values"],
                result["sorted_y_values"] - np.nanmax(result["sorted_y_values"]),
            )
            ax.scatter(
                result["refined_maxima"],
                refined_maxima_y_values,
                marker="*",
                zorder=5,
                s=125,
            )  # , label=f'${key}$ Refined Local Maxima')
            ax.axhline(y=-1.92, linestyle="-", c=current_color, lw=1)
        # Settings and labels
        ax.set_title(
            "Profile likelihood and higher order adjustments",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Z(m)", fontsize=14, fontweight="bold")
        ax.set_ylabel("l", fontsize=14, fontweight="bold")
        true_value, mle_value = lines["true"], lines["mle"]
        sup95, inf95 = lines["sup95"], lines["inf95"]
        # Plotting lines
        ax.axvline(x=true_value, color="w", linestyle="--", lw=3, label="True value")
        ax.axvline(x=mle_value, color="orange", linestyle="-.", lw=3, label="MLE")
        ax.axvline(x=inf95, color="r", linestyle="-.", lw=1, label="Monte-Carlo 95% CL")
        ax.axvline(x=sup95, color="r", linestyle="-.", lw=1)
        # Ticks font sizes and weights
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=8,
            labelcolor="white" if mode == "dark" else "black",
            width=1.5,
        )
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(12)
            tick.set_fontweight("bold")
        # Legend
        ax.legend(loc="upper right", fontsize=16)
        # Remove background color
        # ax.set_facecolor('none')
        # fig.patch.set_facecolor('none')
        plt.tight_layout()
        plt.show()


# Define a new function to plot the r statistic for each curve
def plot_multi_curve_r_statistic(lines, results, mode="dark"):
    """
    Plots multiple curves and their corresponding r statistic on the same plot.
    Parameters:
    - results (dict): Dictionary of results from multi_curve_refine function.
    - mode (str): Mode for the plot's appearance ('dark' or 'white').
    Returns:
    - A plot displaying multiple curves and their r statistics.
    """
    if mode == "dark":
        bg_color = "dark_background"
        title_color = "white"
        label_color = "white"
        grid_color = "gray"
    elif mode == "white":
        bg_color = "default"  # plt.style.use('default')
        title_color = "black"
        label_color = "black"
        grid_color = "gray"
    else:
        raise ValueError("Invalid mode. Accepts either 'dark' or 'white'.")

    with plt.rc_context(
        {
            "axes.edgecolor": grid_color,
            "xtick.color": label_color,
            "ytick.color": label_color,
            "figure.facecolor": grid_color,
        }
    ):
        plt.figure(figsize=(12, 8))
        plt.style.use(bg_color)
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))
        for ax, title in zip(axs, ["Profile likelihood", "r Statistic"]):
            for key, result in results.items():
                # Compute r statistic
                l_max = np.nanmax(result["sorted_y_values"])
                if title == "Profile likelihood":
                    (line,) = ax.plot(
                        result["sorted_x_values"],
                        result["sorted_y_values"] - l_max,
                        label=f"${key}$",
                        linewidth=2,
                    )
                    ax.set_ylim([-100, 10])
                else:
                    mle = result["sorted_x_values"][
                        np.nanargmax(result["sorted_y_values"])
                    ]
                    r_statistic = np.sign(mle - result["sorted_x_values"]) * np.sqrt(
                        2 * (l_max - result["sorted_y_values"])
                    )
                    (line,) = ax.plot(
                        result["sorted_x_values"],
                        r_statistic,
                        label=f"r(${key}$)",
                        linewidth=2,
                    )
                    ax.set_ylim([-100, 100])
                current_color = line.get_color()
                ax.set_yscale("symlog")
                refined_maxima_y_values = np.interp(
                    result["refined_maxima"],
                    result["sorted_x_values"],
                    result["sorted_y_values"] - l_max,
                )
                ax.scatter(
                    result["refined_maxima"],
                    refined_maxima_y_values,
                    marker="*",
                    zorder=5,
                    s=125,
                    color=current_color,
                )
                # Horizontal lines for r statistic
                if title == "r Statistic":
                    ax.axhline(y=1.96, linestyle="-", color="red", linewidth=1)
                    ax.axhline(y=-1.96, linestyle="-", color="red", linewidth=1)

            # Settings and labels
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Z(m)", fontsize=14, fontweight="bold")
            ax.set_ylabel("Value", fontsize=14, fontweight="bold")
            true_value, mle_value = lines["true"], lines["mle"]
            sup95, inf95 = lines["sup95"], lines["inf95"]

            # Plotting lines
            ax.axvline(
                x=true_value, color="w", linestyle="--", lw=3, label="True value"
            )
            ax.axvline(x=mle_value, color="orange", linestyle="-.", lw=3, label="MLE")
            ax.axvline(
                x=inf95, color="r", linestyle="-.", lw=1, label="Monte-Carlo 95% CL"
            )
            ax.axvline(x=sup95, color="r", linestyle="-.", lw=1)

            # Ticks font sizes and weights
            ax.tick_params(
                axis="both",
                which="major",
                labelsize=8,
                labelcolor="white" if mode == "dark" else "black",
                width=1.5,
            )
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontsize(12)
                tick.set_fontweight("bold")

            # Legend
            ax.legend(loc="upper right", fontsize=16)

        plt.tight_layout()
        plt.show()


# Test the new function with sample data


# Function to plot Fisher Information Matrix and Standard Deviations
from matplotlib.colors import LogNorm


# Function to plot Fisher Information Matrix and Standard Deviations with log scale for Fisher Information
def plot_fisher_info(param_dict, fisher_matrix, transformed=None):
    # Calculate the inverse of the Fisher Information Matrix using pseudo-inverse

    if transformed:
        J = []
        for key in param_dict.keys():
            if transformed[key]:
                J.append(np.exp(param_dict[key]) - 1e-18)
            else:
                J.append(1)
        J = np.diag(J)
        fisher_matrix = J.T @ fisher_matrix @ J

    inv_fisher = np.linalg.pinv(fisher_matrix)
    cond_number = np.linalg.cond(fisher_matrix)
    # Calculate the Cramér-Rao lower bound (standard deviation) for each parameter
    std_devs = np.sqrt(inv_fisher)
    # Create subplots with dark background
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plt.style.use("dark_background")
    # Add condition number as text annotation
    annotation = f"Condition Number: {cond_number:.2e}"
    fig.text(
        0.5, 0.05, annotation, ha="center", fontsize=14, fontweight="bold", color="g"
    )

    # Plot Fisher Information Matrix with log scale
    cax1 = axes[0].matshow(fisher_matrix, cmap="gnuplot2", norm=LogNorm())
    axes[0].set_title(
        "Fisher Information Matrix (Log Scale)", fontsize=16, fontweight="bold"
    )
    axes[0].set_xticks(np.arange(len(param_dict)))
    axes[0].set_yticks(np.arange(len(param_dict)))
    axes[0].set_xticklabels(param_dict.keys(), fontsize=12, fontweight="bold")
    axes[0].set_yticklabels(param_dict.keys(), fontsize=12, fontweight="bold")

    # Annotate each cell with the numeric value, formatted as needed
    for i in range(fisher_matrix.shape[0]):
        for j in range(fisher_matrix.shape[1]):
            value = fisher_matrix[i, j]
            text_color = (
                "w"
                if value
                < np.sqrt(np.nanmax(fisher_matrix) + np.nanmin(fisher_matrix)) / 2
                else "k"
            )
            axes[0].text(
                j,
                i,
                f"{value:.0e}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )

    cbar1 = fig.colorbar(cax1, ax=axes[0], fraction=0.046, pad=0.04)
    # Plot Standard Deviations
    cax2 = axes[1].matshow(std_devs, cmap="gnuplot2")
    axes[1].set_title(
        "Standard Deviations (Cramér-Rao lower bound)", fontsize=16, fontweight="bold"
    )
    axes[1].set_xticks(np.arange(len(param_dict)))
    axes[1].set_yticks(np.arange(len(param_dict)))
    axes[1].set_xticklabels(param_dict.keys(), fontsize=12, fontweight="bold")
    axes[1].set_yticklabels(param_dict.keys(), fontsize=12, fontweight="bold")

    # Annotate each cell with the numeric value, formatted as needed
    for i in range(fisher_matrix.shape[0]):
        for j in range(fisher_matrix.shape[1]):
            value = std_devs[i, j]
            if not np.isnan(value):
                text_color = (
                    "w"
                    if value < (np.nanmax(std_devs) + np.nanmin(std_devs)) / 2
                    else "k"
                )
                axes[1].text(
                    i,
                    j,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                )
            elif inv_fisher[i, j] < 0:
                axes[1].text(
                    i, j, "nan", ha="center", va="center", color="w", fontsize=10
                )

    cbar2 = fig.colorbar(cax2, ax=axes[1], fraction=0.046, pad=0.04)
    plt.show()


# Example usage with log scale for Fisher Information
def plot_histograms(results, theta_true=None, bins=20, title_suffix=""):
    """
    Plot histograms for each key in the results dictionary.

    Parameters:
    - results: Dictionary containing arrays of optimized parameters for each observed spectrum
    - theta_true: Dictionary containing true values for comparison
    - bins: Number of bins for the histogram
    - title_suffix: Suffix to add to the title of each subplot
    """

    num_keys = len(results.keys())
    num_cols = min(4, num_keys)  # Limiting number of columns to 3 for readability
    num_rows = int(np.ceil(num_keys / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    # If only one subplot, make sure axes is an array
    if num_keys == 1:
        axes = np.array([axes])

    for ax, (key, values) in zip(axes.flatten(), results.items()):
        ax.hist(values, bins=bins, color="purple", alpha=0.7, edgecolor="black")
        ax.set_title(f"{key} {title_suffix}")

        # Calculate and annotate mean and standard deviation
        mean_value = np.mean(values)
        std_dev = np.std(values)
        textstr = f"Mean: {mean_value:.2f}\nStd Dev: {std_dev:.2f}"

        # If theta_true is provided, plot vertical lines and calculate bias
        if theta_true and key in theta_true:
            true_value = theta_true[key]
            ax.axvline(true_value, color="red", linestyle="dashed", linewidth=1)
            bias = mean_value - true_value
            textstr += f"\nTrue Value: {true_value:.2f}\nBias: {bias:.2f}"

        # These are matplotlib.patch.Patch properties
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

        # Place a text box in upper left in axes coords
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=props,
        )

    # Remove any unused subplots
    for ax in axes.flatten()[num_keys:]:
        ax.remove()

    plt.tight_layout()
    plt.show()


def plot_adjusted_profiles_centered(
    adjusted_profiles,
    psi_points,
    success_messages,
    mle_values,
    l_max_values,
    true_values,
    x_scale=1,
    y_scale=1,
):
    plt.style.use("dark_background")

    n = len(adjusted_profiles.keys())
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1 and cols == 1:
        axs = [plt.gca()]
    else:
        axs = axs.flatten()

    for i, key in enumerate(adjusted_profiles.keys()):
        ax = axs[i]

        # Extract values from the adjusted_profiles dictionary
        lp = np.array(adjusted_profiles[key]["lp"]) - l_max_values[key]
        la = np.array(adjusted_profiles[key]["la"]) - l_max_values[key]
        lm_d = np.array(adjusted_profiles[key]["lm_d"]) - l_max_values[key]
        lm_s = np.array(adjusted_profiles[key]["lm_s"]) - l_max_values[key]
        psi = np.array(psi_points[key])
        success = np.array(success_messages[key])

        mle = mle_values[key]
        true_val = true_values[key]

        # Center around the midpoint between the MLE and true value
        mid_point = (mle + true_val) / 2
        half_width_max = (np.max(psi) - mid_point) / x_scale[key]
        half_width_min = (mid_point - np.min(psi)) / x_scale[key]
        half_width = max(half_width_max, half_width_min)
        x_max = mid_point + half_width
        x_min = mid_point - half_width

        y_min = 1 - (1 - min(lp)) / y_scale[key]
        y_max = min(5, int(100 / y_scale[key]))

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

        # Plot True value
        ax.axvline(true_val, linestyle="--", color="magenta", label="True Value")
        # Plot mle and original profile
        ax.axvline(mle, linestyle="--", color="g", label="MLE")
        ax.plot(psi, lp - np.max(lp), label="lp (Profile)", color="g")
        # Plot original profile
        # Plot various profiles
        ax.plot(psi, la - np.max(la), label="la (Adjusted)", color="cyan")
        ax.plot(psi, lm_s - np.max(lm_s), label="lm_1 (Modified)", color="grey")
        ax.plot(psi, lm_d - np.max(lm_d), label="lm_2 (Modified)", color="orange")

        ax.scatter(
            psi[success], lp[success], color="g", marker="+", label="Success", zorder=5
        )
        ax.scatter(
            psi[~success],
            lp[~success],
            color="r",
            marker="+",
            label="Failure",
            zorder=5,
        )
        ax.set_title(f"Profiles for {key}")
        ax.set_xlabel(key)
        ax.set_ylabel(f"log likelihoods")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize="small", loc="upper right")

    plt.tight_layout()
    plt.show()


# plot_adjusted_profiles_centered(adjusted_profiles, sorted_psi_points, sorted_success_messages, mle_values, l_max_values, theta_true, x_scale = 1, y_scale = 1)
# jacob_nuisance.shape
