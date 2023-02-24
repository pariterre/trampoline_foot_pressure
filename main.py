import os

from misc import Data, DataReader, JumpName, JumpDirection, JumpCategory, JumpPosition


# TODO: 2-3 sujets (séparés)
#   # TODO: Trajectoire vitesse et Force
# TODO: Découper en sous-phases (5?) d'impulsion
# TODO: Apprentissage machine si on trouve rien

# ---- OPTIONS ---- #
data_folder = "data"
figure_save_folder = "results/figures"
subjects = ("sujet1",)
forces_black_list_files = (
    "13815_9.23.2022_13.42.17",
    "13818_9.23.2022_15.14.29",
    "13833_9.23.2022_12.51.15",
    "13835_9.24.2022_13.18.26",
    "13842_9.23.2022_12.57.53",
)
# jump_filter = {"category": JumpCategory.BARANI, "position": JumpPosition.STRAIGHT}
# jump_filter = {"category": JumpCategory.BARANI}
# jump_filter = {"position": JumpPosition.TUCK}
# jump_filter = {"name": JumpName.BARANI_STRAIGHT}
jump_filter = {}
jump_sequence = (
    JumpName.BACK_SOMERSAULT_STRAIGHT,
    JumpName.BARANI_STRAIGHT,
    JumpName.BACK_SOMERSAULT_TUCK,
    JumpName.BARANI_TUCK,
    JumpName.BACK_SOMERSAULT_TUCK,
    JumpName.BARANI_TUCK,
    JumpName.BACK_SOMERSAULT_STRAIGHT,
    JumpName.BARANI_STRAIGHT,
)
show_cop = False
show_y_movement = False
show_force_velocity = True
skip_huge_files = False
save_figures = True
# ----------------- #


def main():

    if show_force_velocity and skip_huge_files:
        raise RuntimeError("'show_force_velocity' requires to 'skip_huge_files' to be 'False'")

    if save_figures:
        if not os.path.exists(figure_save_folder):
            os.makedirs(figure_save_folder)

    for subject in subjects:
        folder = f"{data_folder}/{subject}"
        filenames = DataReader.fetch_trial_names(folder)

        cycl_data = []
        force_data = []
        for filename in filenames:
            if not skip_huge_files and filename in forces_black_list_files:
                continue
            # Load data
            cycl_data.append(
                DataReader.read_cycl_data(f"{data_folder}/{subject}/{filename}", jump_sequence=jump_sequence)
            )
            force_data.append(
                DataReader.read_sensor_data(f"{data_folder}/{subject}/{filename}", jump_sequence=jump_sequence)
                if not skip_huge_files else None
            )

        # Extract some data
        backward_jumps_cycl = []
        frontward_jumps_cycl = []
        backward_jumps_forces = []
        frontward_jumps_forces = []
        for data in cycl_data:
            backward_jumps_cycl.extend(data.filtered_data(direction=JumpDirection.BACKWARD, **jump_filter))
            frontward_jumps_cycl.extend(data.filtered_data(direction=JumpDirection.FRONTWARD, **jump_filter))
        if not skip_huge_files:
            for data in force_data:
                backward_jumps_forces.extend(data.filtered_data(direction=JumpDirection.BACKWARD, **jump_filter))
                frontward_jumps_forces.extend(data.filtered_data(direction=JumpDirection.FRONTWARD, **jump_filter))

        # Print if required
        # Concatenated the data in a single matrix to ease printing
        for jumps_cycl, jumps_forces in zip([backward_jumps_cycl, frontward_jumps_cycl], [backward_jumps_forces, frontward_jumps_forces]):
            jumps_flight_times = [jump.flight_times[0] for jump in jumps_cycl]
            if not jumps_flight_times:
                continue  # If there is no frontward or backward jump in the filtered data

            condition_name = f"{jumps_cycl[0].jump_sequence[0].value[0].value}"
            if jump_filter is not None:
                for key in jump_filter.keys():
                    condition_name += f", {key}: {jump_filter[key].name}"
                condition_name += f" only"

            min_flight_times = min(jumps_flight_times)
            max_flight_times = max(jumps_flight_times)
            range_flight_times = max_flight_times - min_flight_times

            if show_cop:
                fig = None
                fig_name = f"CoP ({condition_name})"
                for i, jump in enumerate(jumps_cycl):
                    green_shift = (jump.flight_times[0] - min_flight_times) / range_flight_times
                    fig = jump.plot(
                        figure=fig_name,
                        fig=fig,
                        title=f"CoP ({condition_name})\nLonger flight time green shifted",
                        x_label="X-coordinates (m)",
                        y_label="Y-coordinates (m)",
                        color=(1-green_shift, green_shift, 0, 1),
                    )
                if save_figures:
                    fig[0].set_size_inches(16, 9)
                    fig[0].savefig(f"{figure_save_folder}/CoP_{condition_name}.png", dpi=300)

            if show_y_movement:
                fig = None
                fig_name = f"Anteroposterior ({condition_name})"
                for i, jump in enumerate(jumps_cycl):
                    green_shift = (jump.flight_times[0] - min_flight_times) / range_flight_times
                    fig = jump.plot(
                        override_x=jump.t - jump.t[0],
                        override_y=jump.y[:, 1],
                        figure=fig_name,
                        fig=fig,
                        title=f"CoP ({condition_name})\nLonger flight time green shifted",
                        x_label="Temps (s)",
                        y_label="Anteroposterior movement (m)",
                        color=(1-green_shift, green_shift, 0, 1),
                    )
                if save_figures:
                    fig[0].set_size_inches(16, 9)
                    fig[0].savefig(f"{figure_save_folder}/anteroposterior_{condition_name}.png", dpi=300)

            if show_force_velocity:
                fig = None
                fig_name = f"Forces ({condition_name})"
                for i, (jump_cycl, jump_forces) in enumerate(zip(jumps_cycl, jumps_forces)):
                    green_shift = (jump_cycl.flight_times[0] - min_flight_times) / range_flight_times
                    fig = jump_forces.plot(
                        override_x=jump_forces.t - jump_forces.t[0],
                        figure=fig_name,
                        fig=fig,
                        title=f"Forces ({condition_name})\nLonger flight time green shifted",
                        x_label="Temps (s)",
                        y_label="Forces (N)",
                        color=(1 - green_shift, green_shift, 0, 1),
                    )
                if save_figures:
                    fig[0].set_size_inches(16, 9)
                    fig[0].savefig(f"{figure_save_folder}/forces_{condition_name}.png", dpi=300)

        if show_cop or show_y_movement or show_force_velocity:
            Data.show()


if __name__ == "__main__":
    main()
