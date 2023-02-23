from matplotlib import pyplot as plt
import numpy as np

from .helpers import JumpType, JumpDirection, PhaseType


class Data:
    def __init__(
        self, data=None, jump_sequence: tuple[JumpType, ...] = None, nb_sensors: int = 0, conversion_factor: float = 1
    ):
        """
        Create a Data structure with 't' as time vector holder and 'y' as data holder
        Parameters
        ----------
        data
            Data to copy from
        jump_sequence
            The expected sequence of jumps in the trial
        nb_sensors
            The number of sensors (column) in the 'y' data holder
        conversion_factor
            The factor to convert the data when using the 'append' method
        """

        if data is not None:
            self.t = data.t
            self.y = data.y
            self.conversion_factor = data.conversion_factor

            # Compute other stuff if they happen to be empty
            takeoffs_indices, landing_indices = self.compute_timings_indices(
                np.sum(self.y, axis=1)[:, np.newaxis]
            )
            self.takeoffs_indices = takeoffs_indices if data.takeoffs_indices.shape[0] == 0 else data.takeoffs_indices
            self.landings_indices = landing_indices if data.landings_indices.shape[0] == 0 else data.landings_indices
            self.jump_sequence = jump_sequence if jump_sequence is not None else data.jump_sequence
        else:
            self.t: np.ndarray = np.ndarray((0,))
            self.y: np.ndarray = np.ndarray((0, nb_sensors))
            self.conversion_factor = conversion_factor
            self.takeoffs_indices, self.landings_indices = self.compute_timings_indices(
                np.sum(self.y, axis=1)[:, np.newaxis]
            )
            self.jump_sequence = jump_sequence

        # Sanity check
        if self.jump_sequence is not None and (
            sum(np.isfinite(self.landings_indices)) != len(self.jump_sequence) or sum(np.isfinite(self.takeoffs_indices)) != len(self.jump_sequence)
        ):
            raise RuntimeError(
                "The number of jumps in the trials does not correspond to the number of jumps in the provided sequence"
            )

    def append(self, t, y) -> None:
        """
        Add data to the data set

        Parameters
        ----------
        t
            The time to add
        y
            The data to add (converted with self.conversion_factor)
        """

        self.t = np.concatenate((self.t, (t,)))

        # Remove the MAX_INT and convert to m
        y = [data * self.conversion_factor if data != 2147483647 else np.nan for data in y]
        self.y = np.concatenate((self.y, (y,)))

    def concatenate(self, other):
        """
        Concatenate a data set to another, assuming the time of self is added as an offset to other

        Parameters
        ----------
        other
            The data to concatenate

        Returns
        -------
        The concatenated data
        """

        out = Data(data=self, jump_sequence=self.jump_sequence)
        time_offset = out.t[-1]

        previous_t = out.t
        previous_takeoffs_indices = out.takeoffs_indices
        out.t = np.concatenate((out.t, time_offset + other.t))
        out.y = np.concatenate((out.y, other.y))
        out.landings_indices = np.concatenate(
            (
                out.landings_indices[np.isfinite(out.landings_indices)],
                other.landings_indices[np.isfinite(other.landings_indices)] + previous_t.shape[0]
            )
        )
        out.takeoffs_indices = np.concatenate(
            (
                out.takeoffs_indices[np.isfinite(out.takeoffs_indices)],
                other.takeoffs_indices[np.isfinite(other.takeoffs_indices)] + previous_t.shape[0]
            )
        )

        # Take into account the case of only mat indices are presents
        if np.isnan(previous_takeoffs_indices[0]):
            out.takeoffs_indices = np.concatenate(((np.nan, ), out.takeoffs_indices))
        if np.isnan(other.landings_indices[-1]):
            out.landings_indices = np.concatenate((out.landings_indices, (np.nan, )))

        out.jump_sequence = self.jump_sequence + other.jump_sequence
        # Sanity check
        if sum(np.isfinite(self.landings_indices)) != len(self.jump_sequence) or sum(np.isfinite(self.takeoffs_indices)) != len(self.jump_sequence):
            raise RuntimeError(
                "The number of jumps in the trials does not correspond to the number of jumps in the provided sequence"
            )

        return out

    @property
    def flight_indices(self) -> tuple[tuple[int, int], ...]:
        """
        Return the indices of flights in the order of takeoff/landing
        """
        return tuple(zip(np.array(self.takeoffs_indices, dtype=np.int64), np.array(self.landings_indices, dtype=np.int64)))

    @property
    def flight_times(self) -> tuple[float, ...]:
        """
        Get the times in the air
        """
        return tuple(self.t[landing - 1] - self.t[takeoff] for takeoff, landing in self.flight_indices)

    @property
    def mat_indices(self) -> tuple[tuple[int, int], ...]:
        """
        Return the indices of landed in the order of landing/takeoff
        """
        return tuple(zip(np.array(self.landings_indices[:-1], dtype=np.int64), np.array(self.takeoffs_indices[1:], dtype=np.int64)))

    @property
    def mat_times(self) -> tuple[float, ...]:
        """
        Get the times in the mat
        """
        return tuple(self.t[takeoff - 1] - self.t[landing] for landing, takeoff in self.mat_indices)

    def filtered_data(
        self, phase: PhaseType, direction: JumpDirection = None, jump_type: JumpType = None
    ) -> tuple["Data", ...]:
        """
        Filters the data

        Parameters
        ----------
        phase
            The phase (on mat or in air) to extract
        direction
            The direction of jump to return. If 'None', all JumpDirection are returned
        jump_type
            The type of jump to return. If 'None', all JumpType are returned

        Returns
        -------
        A tuple of tuples all the jumps data that corresponds to the criteria. The type of the data is the same as
        the original one
        """
        if phase == PhaseType.MAT:
            all_indices = self.mat_indices
        elif phase == PhaseType.AERIAL:
            all_indices = self.flight_indices
        else:
            raise ValueError("Wrong value for phase")

        out = []
        for jump, indices in zip(self.jump_sequence, all_indices):
            if jump_type is not None and jump != jump_type:
                continue
            if direction is not None and jump.value != direction:
                continue

            tp = Data(conversion_factor=self.conversion_factor)
            start, finish = indices
            tp.t = self.t[start:finish]
            tp.y = self.y[start:finish, :]
            tp.jump_sequence = (jump,)
            if phase == PhaseType.MAT:
                tp.takeoffs_indices = np.array((np.nan, tp.y.shape[0]))
                tp.landings_indices = np.array((0, np.nan))
            elif phase == PhaseType.AERIAL:
                tp.takeoffs_indices = np.array((0, ))
                tp.landings_indices = np.array((tp.y.shape[0], ))
            else:
                raise ValueError("Wrong value for phase")
            tp = type(self)(tp)  # Up cast the data to the same format as self
            out.append(tp)
        return tuple(out)

    def plot(
        self,
        override_y: np.ndarray = None,
        **figure_options,
    ) -> plt.figure:
        """
        Plot the data as time dependent variables

        Parameters
        ----------
        override_y
            Force to plot this y data instead of the one in the self.y attribute
        figure_options
            see _prepare_figure inputs

        Returns
        -------
        The matplotlib figure handler if show_now was set to False
        """

        fig, ax, color, show_now = self._prepare_figure(**figure_options)

        ax.plot(self.t, override_y if override_y is not None else self.y, color=color)

        if show_now:
            plt.show()

        return fig if not show_now else None

    def plot_flight_times(self, factor: float = 1, **figure_options) -> plt.figure:
        """
        Plot the flight times as constant values of the flight period

        Parameters
        ----------
        figure_options
            see _prepare_figure inputs
        factor
            Proportional factor

        Returns
        -------
        The matplotlib figure handler if show_now was set to False
        """
        y = np.nan * np.ndarray(self.y.shape[0])

        for (takeoff, landing), flight in zip(self.flight_indices, self.flight_times):
            y[takeoff:landing] = flight
        return Data.plot(self, override_y=y * factor, **figure_options)

    def plot_mat_times(self, factor: float = 1, **figure_options) -> plt.figure:
        """
        Plot the nat times as constant values of the mat period

        Parameters
        ----------
        figure_options
            see _prepare_figure inputs
        factor
            Proportional factor

        Returns
        -------
        The matplotlib figure handler if show_now was set to False
        """
        y = np.nan * np.ndarray(self.y.shape[0])

        for (takeoff, landing), landed in zip(self.mat_indices, self.mat_times):
            y[takeoff:landing] = landed
        return Data.plot(self, override_y=y * factor, **figure_options)

    @staticmethod
    def show() -> None:
        """
        Just a convenient method so one does not have to include matplotlib in other script just to call plt.show()
        """
        plt.show()

    @staticmethod
    def _prepare_figure(
        figure: str = None,
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        x_lim: list[float, float] = None,
        y_lim: list[float, float] = None,
        color: str = None,
        axis_on_right: bool = False,
        show_now: bool = False,
        maximize: bool = False,
    ) -> tuple[plt.figure, plt.axis, str, bool]:
        """

        Parameters
        ----------
        figure
            The name of the figure. If two figures has the same name, they are drawn on the same graph
        title
            The title of the figure
        x_label
            The name of the X-axis
        y_label
            The name of the Y-axis
        x_lim
            The limits of the X-axis
        y_lim
            The limits of the Y-axis
        color
            The color of the plot
        show_now
            If the plot should be shown right now (blocking)
        maximize
            If the figure should be maximized in the screen
        Returns
        -------
        The figure and the axis handler
        """

        fig = plt.figure(figure)

        if not fig.axes:
            ax = plt.axes()
        else:
            ax = fig.axes[-1]
            if axis_on_right:
                ax = ax.twinx()

        if title is not None:
            ax.set_title(title)
        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)
        if x_lim is not None:
            ax.set_ylim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        if axis_on_right:
            ax.yaxis.tick_right()

        if maximize:
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()

        return fig, ax, color, show_now

    @staticmethod
    def compute_timings_indices(data) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the flight time for each flights in the data.
        The flight moments are defined as "nan" in the displacement data.

        Parameters
        ----------
        data
            Data vector to compute the timings time from

        Returns
        -------
        The timing indices of the jumps (takeoff and landing)
        """

        if not data.any():
            return np.ndarray((0,)), np.ndarray((0,))

        # Find all landing and takeoff indices
        currently_in_air = 1 * np.isnan(data)  # 1 for True, 0 for False
        padding = ((0,),)
        events = np.concatenate((padding, currently_in_air[1:] - currently_in_air[:-1]))
        events[:2] = 0  # Remove any possible artifact from cop_displacement starting
        landings_indices = np.where(events == -1)[0]
        takeoffs_indices = np.where(events == 1)[0]

        if not takeoffs_indices.any() and not landings_indices.any():
            # If nothing is found, it is okay
            return takeoffs_indices, landings_indices

        # Remove starting and ending artifacts and perform sanity check
        if landings_indices[0] < takeoffs_indices[0]:
            landings_indices = landings_indices[1:]
        if takeoffs_indices[-1] > landings_indices[-1]:
            takeoffs_indices = takeoffs_indices[:-1]
        if len(takeoffs_indices) != len(landings_indices):
            raise RuntimeError(
                f"The number of takeoffs ({len(takeoffs_indices)} is not equal "
                f"to number of landings {len(landings_indices)}"
            )

        return takeoffs_indices, landings_indices
