import os

from misc import Data, DataReader, concatenate_data, JumpType, PhaseType, JumpDirection
from scipy.stats import pearsonr


# TODO: Superposer CoP pour sauts avant et arrière
# TODO: 2-3 sujets (séparés)
#   # TODO: Trajectoire en antéropostérieur (contre temps) et vitesse et Force
#   # TODO: Code couleur si gain/perte de hauteur (vert à rouge)
# TODO: Découper en sous-phases (5?) d'impulsion
# TODO: Apprentissage machine si on trouve rien


def main():
    # ---- OPTIONS ---- #
    data_folder = "data"
    figure_save_folder = "results/figures"
    subjects = ("sujet1",)
    jump_sequence = (
        JumpType.BACK_SOMERSAULT_STRAIGHT,
        JumpType.BARANI_STRAIGHT,
        JumpType.BACK_SOMERSAULT_TUCK,
        JumpType.BARANI_TUCK,
        JumpType.BACK_SOMERSAULT_TUCK,
        JumpType.BARANI_TUCK,
        JumpType.BACK_SOMERSAULT_STRAIGHT,
        JumpType.BARANI_STRAIGHT,
    )
    show_cop = False
    show_cop_displacement = True
    show_cop_velocity = False
    show_cop_acceleration = False
    show_sensors = False
    skip_huge_files = True
    save_figures = False
    # ----------------- #

    if show_sensors and skip_huge_files:
        raise ValueError("It is not possible to 'show_sensors' if 'skip_huge_files'")

    if save_figures:
        if not os.path.exists(figure_save_folder):
            os.makedirs(figure_save_folder)

    for subject in subjects:
        folder = f"{data_folder}/{subject}"
        filenames = DataReader.fetch_trial_names(folder)

        cycl_data = []
        force_data = []
        for filename in filenames:
            # Load data
            cycl_data.append(
                DataReader.read_cycl_data(f"{data_folder}/{subject}/{filename}", jump_sequence=jump_sequence)
            )
            force_data.append(
                DataReader.read_sensor_data(f"{data_folder}/{subject}/{filename}", jump_sequence=jump_sequence) if not skip_huge_files else None
            )

        # Concatenated the data in a single matrix
        cycl_data = concatenate_data(cycl_data)
        force_data = concatenate_data(force_data) if not skip_huge_files else None

        backward_jumps = cycl_data.filtered_data(phase=PhaseType.MAT, direction=JumpDirection.BACKWARD)

        # Print if required
        filename = "All data"
        if show_cop:
            fig_name = f"CoP ({filename})"
            fig = cycl_data.plot(
                figure=fig_name,
                title="CoP",
                x_label="X-coordinates (m)",
                y_label="Y-coordinates (m)",
                color="blue",
            )
            if save_figures:
                fig.set_size_inches(16, 9)
                fig.savefig(f"{figure_save_folder}/CoP.png", dpi=300)

        if show_cop_displacement:
            fig_name = f"CoP displacement ({filename})"
            fig = cycl_data.plot_displacement(
                figure=fig_name,
                title=f"CoP displacement (blue) and Jump time (orange)\n"
                f"Integral correlation = {pearsonr(cycl_data.displacement_integral, cycl_data.flight_times[1:])[0]:0.3f}\n"
                f"Ranges correlation = {pearsonr(cycl_data.displacement_ranges, cycl_data.flight_times[1:])[0]:0.3f}\n",
                x_label="Time (s)",
                y_label="CoP displacement (m)",
                color="blue",
            )
            cycl_data.plot_flight_times(
                figure=fig_name,
                y_label="Jump time (s)",
                axis_on_right=True,
                color="orange",
            )
            if save_figures:
                fig.set_size_inches(16, 9)
                fig.savefig(f"{figure_save_folder}/CoP_displacement.png", dpi=300)

        if show_cop_velocity:
            fig_name = f"CoP Velocity ({filename})"
            fig = cycl_data.plot_velocity(
                figure=fig_name,
                title=f"CoP velocity (blue) and Jump time (orange)\n"
                f"Integral correlation = {pearsonr(cycl_data.velocity_integral, cycl_data.flight_times[1:])[0]:0.3f}\n"
                f"Ranges correlation = {pearsonr(cycl_data.velocity_ranges, cycl_data.flight_times[1:])[0]:0.3f}\n",
                x_label="Time (s)",
                y_label="CoP velocity (m/s)",
                color="blue",
            )
            cycl_data.plot_flight_times(
                figure=fig_name,
                y_label="Jump time (s)",
                axis_on_right=True,
                color="orange",
            )
            if save_figures:
                fig.set_size_inches(16, 9)
                fig.savefig(f"{figure_save_folder}/CoP_velocity.png", dpi=300)

        if show_cop_acceleration:
            fig_name = f"CoP Acceleration ({filename})"
            fig = cycl_data.plot_acceleration(
                figure=fig_name,
                title=f"CoP acceleration (blue) and Jump time (orange)\n"
                f"Integral correlation = {pearsonr(cycl_data.acceleration_integral, cycl_data.flight_times[1:])[0]:0.3f}\n"
                f"Ranges correlation = {pearsonr(cycl_data.acceleration_ranges, cycl_data.flight_times[1:])[0]:0.3f}\n",
                x_label="Time (s)",
                y_label="CoP acceleration (m/s/s)",
                y_lim=[-60, 200],
                color="blue",
            )
            cycl_data.plot_flight_times(
                figure=fig_name,
                y_label="Jump time (s)",
                y_lim=[1, 2],
                axis_on_right=True,
                color="orange",
            )
            if save_figures:
                fig.set_size_inches(16, 9)
                fig.savefig(f"{figure_save_folder}/CoP_acceleration.png", dpi=300)

        if show_sensors:
            fig_name = f"Forces ({filename})"
            fig = force_data.plot(
                figure=fig_name,
                title="Sensor forces (blue) and Jump time (orange)\n"
                f"Integral correlation = {pearsonr(force_data.force_integral, force_data.flight_times[1:])[0]:0.3f}\n",
                x_label="Time (s)",
                y_label="Force (N)",
                color="blue",
            )
            force_data.plot_flight_times(
                figure=fig_name,
                y_label="Jump time (s)",
                y_lim=[1, 2],
                axis_on_right=True,
                color="orange",
            )
            if save_figures:
                fig.set_size_inches(16, 9)
                fig.savefig(f"{figure_save_folder}/forces.png", dpi=300)

        if show_cop or show_cop_displacement or show_cop_velocity or show_cop_acceleration or show_sensors:
            Data.show()


if __name__ == "__main__":
    main()
