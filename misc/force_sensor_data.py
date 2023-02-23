import numpy as np

from .data import Data
from .helpers import integral


class ForceSensorData(Data):
    def __init__(self, data: Data, **kwargs):
        data.y = np.sum(data.y, axis=1)[:, np.newaxis]
        data.y[data.y < 20] = np.nan
        super().__init__(data=data, **kwargs)

    @property
    def force_integral(self) -> tuple[float, ...]:
        """
        Get the force integral (impulse) in the mat
        """
        return tuple(
            integral(self.t[l:t], self.y[l:t, :])
            for t, l in zip(self.takeoffs_indices[1:], self.landings_indices[0:-1])
        )
