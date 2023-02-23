import numpy as np
from matplotlib import pyplot as plt

from .data import Data
from .helpers import derivative, integral


class CoPData(Data):
    def __init__(self, data: Data, **kwargs):
        super().__init__(data=data, **kwargs)
        self.displacement = self._compute_cop_displacement(window=2)
        self.velocity = derivative(self.t, self.displacement, window=2)
        self.acceleration = derivative(self.t, self.velocity, window=2)

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
        return CoPData(super().concatenate(other))

    @property
    def displacement_integral(self) -> tuple[float, ...]:
        """
        Get the horizontal displacement integral in the mat
        """
        return tuple(
            integral(self.t[l:t], self.displacement[l:t])
            for t, l in zip(self.takeoffs_indices[1:], self.landings_indices[0:-1])
        )

    @property
    def displacement_ranges(self) -> tuple[float, ...]:
        """
        Get the horizontal range
        """
        return tuple(
            np.nanmax(self.displacement[l:t]) - np.nanmin(self.displacement[l:t])
            for t, l in zip(self.takeoffs_indices[1:], self.landings_indices[0:-1])
        )

    @property
    def velocity_integral(self) -> tuple[float, ...]:
        """
        Get the horizontal impulses in the mat
        """
        return tuple(
            integral(self.t[l:t], self.velocity[l:t])
            for t, l in zip(self.takeoffs_indices[1:], self.landings_indices[0:-1])
        )

    @property
    def velocity_ranges(self) -> tuple[float, ...]:
        """
        Get the horizontal range
        """
        return tuple(
            np.nanmax(self.velocity[l:t]) - np.nanmin(self.velocity[l:t])
            for t, l in zip(self.takeoffs_indices[1:], self.landings_indices[0:-1])
        )

    @property
    def acceleration_integral(self) -> tuple[float, ...]:
        """
        Get the horizontal acceleration integral in the mat
        """
        return tuple(
            integral(self.t[l:t], self.acceleration[l:t])
            for t, l in zip(self.takeoffs_indices[1:], self.landings_indices[0:-1])
        )

    @property
    def acceleration_ranges(self) -> tuple[float, ...]:
        """
        Get the horizontal range
        """
        return tuple(
            np.nanmax(self.acceleration[l:t]) - np.nanmin(self.acceleration[l:t])
            for t, l in zip(self.takeoffs_indices[1:], self.landings_indices[0:-1])
        )

    def plot(
        self,
        override_y: np.ndarray = None,
        **figure_options,
    ) -> plt.figure:
        """
        Plot the data as XY position in an axis('equal') manner

        Parameters
        ----------
        override_y
            Force to plot this y data instead of the self.y attribute
        figure_options
            see _prepare_figure inputs

        Returns
        -------
        The matplotlib figure handler if show_now was set to False
        """

        fig, ax, color, show_now = self._prepare_figure(**figure_options)

        ax.plot(self.y[:, 0], self.y[:, 1], color=color)
        ax.axis("equal")

        if show_now:
            plt.show()

        return fig if not show_now else None

    def plot_displacement(self, **figure_options) -> plt.figure:
        """
        Plot the CoP displacement against time

        Parameters
        ----------
        figure_options
            see _prepare_figure inputs

        Returns
        -------
        The matplotlib figure handler if show_now was set to False
        """

        return super().plot(override_y=self.displacement, **figure_options)

    def plot_velocity(self, **figure_options) -> plt.figure:
        """
        Plot the CoP velocity against time

        Parameters
        ----------
        figure_options
            see _prepare_figure inputs

        Returns
        -------
        The matplotlib figure handler if show_now was set to False
        """

        return super().plot(override_y=self.velocity, **figure_options)

    def plot_acceleration(self, **figure_options) -> plt.figure:
        """
        Plot the CoP acceleration against time

        Parameters
        ----------
        figure_options
            see _prepare_figure inputs

        Returns
        -------
        The matplotlib figure handler if show_now was set to False
        """

        return super().plot(override_y=self.acceleration, **figure_options)

    def _compute_cop_displacement(self, window: int = 1) -> np.ndarray:
        """
        Compute the CoP displacement
        Parameters
        ----------
        window
            The window to perform the filtering on

        Returns
        -------

        """

        two_windows = window * 2
        padding = np.nan * np.zeros((window, 1))

        return np.concatenate(
            (
                padding,
                np.linalg.norm(self.y[two_windows:, :] - self.y[:-two_windows, :], axis=1)[:, np.newaxis],
                padding,
            )
        )
