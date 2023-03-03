from typing import Any

from matplotlib import pyplot as plt
import numpy as np

from .helpers import JumpName, JumpDirection, JumpPosition, JumpCategory


class Data:
    def __init__(
        self, data=None, jump_sequence: tuple[JumpName, ...] = None, nb_sensors: int = 0, conversion_factor: float = 1
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
            self.landings_indices.shape[0] != len(self.jump_sequence) + 1 or self.takeoffs_indices.shape[0] != len(self.jump_sequence) + 1
        ):
            raise RuntimeError(
                "The number of jumps in the trial does not correspond to the number of jumps in the provided sequence"
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

    @property
    def flight_indices(self) -> tuple[tuple[int, int], ...]:
        """
        Return the indices of flights in the order of takeoff/landing
        """
        return tuple(zip(self.takeoffs_indices[:-1], self.landings_indices[1:]))

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
        return tuple(zip(self.landings_indices, self.takeoffs_indices))

    @property
    def mat_times(self) -> tuple[float, ...]:
        """
        Get the times in the mat
        """
        # Last pair of indices are for the final landing which has no time
        return tuple(self.t[takeoff - 1] - self.t[landing] for landing, takeoff in self.mat_indices[:-1])

    def filtered_data(
        self,
        direction:
        JumpDirection = None,
        name: JumpName = None,
        position: JumpPosition = None,
        category: JumpCategory = None,
    ) -> tuple["Data", ...]:
        """
        Filters the data

        Parameters
        ----------
        direction
            The direction of jump to return. If 'None', all JumpDirection are returned
        name
            The type of jump to return. If 'None', all JumpType are returned
        position
            The position of the jump. If 'None', all positions are returned
        category
            The category of the jump. If 'None', all categories are returned

        Returns
        -------
        A tuple of tuples all the jumps data that corresponds to the criteria. The type of the data is the same as
        the original one
        """
        out = []
        for i, jump in enumerate(self.jump_sequence):
            if name is not None and jump != name:
                continue
            if direction is not None and jump.value[0] != direction:
                continue
            if category is not None and jump.value[1] != category:
                continue
            if position is not None and jump.value[2] != position:
                continue

            tp = Data(conversion_factor=self.conversion_factor)
            start, finish = self.landings_indices[i], self.landings_indices[i + 1]
            tp.t = self.t[start:finish]
            tp.y = self.y[start:finish, :]
            tp.jump_sequence = (jump,)
            tp.landings_indices = np.array((0, tp.y.shape[0]))
            tp.takeoffs_indices = np.array((self.takeoffs_indices[i] - self.landings_indices[i], None))

            out.append(type(self)(tp))  # Up cast the data to the same format as self
        return tuple(out)

    def plot(
        self,
        override_x: np.ndarray = None,
        override_y: np.ndarray = None,
        fig: tuple[plt.figure, plt.axis] = None,
        color: Any = None,
        **figure_options,
    ) -> tuple[plt.figure, plt.axis]:
        """
        Plot the data as time dependent variables

        Parameters
        ----------
        override_x
            Force to plot this x data instead of the self.t attribute
        override_y
            Force to plot this y data instead of the one in the self.y attribute
        fig
            The handlers returned by a previous call of the plot function
        color
            Part of the figure_options
        figure_options
            see _prepare_figure inputs

        Returns
        -------
        The matplotlib figure handler if show_now was set to False
        """

        if fig is None:
            pltfig, ax, color, show_now = self._prepare_figure(color=color, **figure_options)
        else:
            pltfig, ax = fig
            show_now = False

        x = self.t if override_x is None else override_x
        y = self.y if override_y is None else override_y
        ax.plot(x, y, color=color)

        if show_now:
            plt.show()

        return (pltfig, ax) if not show_now else None

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

        if not landings_indices.any() and takeoffs_indices.shape[0] == 1:
            # This can happen if the trial is precisely cut from the landing to next landing of exactly 1 trial
            return takeoffs_indices, landings_indices

        # Remove starting and ending artifacts and perform sanity check
        if landings_indices[0] > takeoffs_indices[0]:  # Trial starts when the subject first hit the mat
            takeoffs_indices = takeoffs_indices[1:]
        if takeoffs_indices[-1] > landings_indices[-1]:  # The trial ends when the subject last hit the mat
            takeoffs_indices = takeoffs_indices[:-1]
        takeoffs_indices = np.concatenate((takeoffs_indices, (None, )))  # Add a None for shape consistency

        if len(takeoffs_indices) != len(landings_indices):
            raise RuntimeError(
                f"The number of takeoffs ({len(takeoffs_indices)} is not equal "
                f"to number of landings {len(landings_indices)}"
            )

        return takeoffs_indices, landings_indices
