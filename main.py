import os

from misc import Data, DataReader, JumpType, JumpDirection


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
    show_cop = True
    skip_huge_files = True
    save_figures = False
    # ----------------- #

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
                DataReader.read_sensor_data(f"{data_folder}/{subject}/{filename}", jump_sequence=jump_sequence)
                if not skip_huge_files else None
            )

        # Extract some data
        backward_jumps = []
        frontward_jumps = []
        for data in cycl_data:
            backward_jumps.extend(data.filtered_data(direction=JumpDirection.BACKWARD))
            frontward_jumps.extend(data.filtered_data(direction=JumpDirection.FRONTWARD))

        # Print if required
        # Concatenated the data in a single matrix to ease printing
        if show_cop:
            for segmented_jumps in [backward_jumps, frontward_jumps]:
                fig = None
                fig_name = f"CoP ({segmented_jumps[0].jump_sequence[0].value})"
                jumps_flight_times = [jump.flight_times[0] for jump in segmented_jumps]
                min_flight_times = min(jumps_flight_times)
                max_flight_times = max(jumps_flight_times)
                range_flight_times = max_flight_times - min_flight_times

                for i, jumps in enumerate(segmented_jumps):
                    green_shift = (jumps.flight_times[0] - min_flight_times) / range_flight_times
                    fig = jumps.plot(
                        figure=fig_name,
                        fig=fig,
                        title=f"CoP ({segmented_jumps[0].jump_sequence[0].value})\nLonger flight time green shifted",
                        x_label="X-coordinates (m)",
                        y_label="Y-coordinates (m)",
                        color=(1-green_shift, green_shift, 0, 1),
                    )
                if save_figures:
                    fig[0].set_size_inches(16, 9)
                    fig[0].savefig(f"{figure_save_folder}/CoP.png", dpi=300)

        if show_cop:
            Data.show()


if __name__ == "__main__":
    main()
