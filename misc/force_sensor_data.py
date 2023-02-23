import numpy as np

from .data import Data
from .helpers import integral


class ForceSensorData(Data):
    def __init__(self, data: Data, **kwargs):
        data.y = np.sum(data.y, axis=1)[:, np.newaxis]
        data.y[data.y < 20] = np.nan
        super().__init__(data=data, **kwargs)

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
        return ForceSensorData(super().concatenate(other))

    @property
    def force_integral(self) -> tuple[float, ...]:
        """
        Get the force integral (impulse) in the mat
        """
        return tuple(
            integral(self.t[l:t], self.y[l:t, :])
            for t, l in zip(self.takeoffs_indices[1:], self.landings_indices[0:-1])
        )
