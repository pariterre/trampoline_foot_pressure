import os
import re

import csv
import numpy as np

from .data import Data
from .cop_data import CoPData
from .force_sensor_data import ForceSensorData
from .helpers import JumpType


class DataReader:
    @staticmethod
    def read_cycl_data(filepath, jump_sequence: tuple[JumpType, ...]) -> CoPData:
        """
        Read the CYCL file which is the CoP coordinates of cyclogramme

        Parameters
        ----------
        filepath
            The path of the file to read
        jump_sequence
            The expected sequence of jumps in the data
        Returns
        -------
        The parsed data in the file
        """

        return CoPData(
            DataReader._read_csv(f"{filepath}_CYCL.CSV", nb_sensors=2, nb_headers_rows=1, conversion_factor=1 / 1000), jump_sequence=jump_sequence
        )

    @staticmethod
    def read_sensor_data(filepath, jump_sequence: tuple[JumpType, ...]) -> ForceSensorData:
        """
        Read the GL file which is the CoP coordinate of gait line

        Parameters
        ----------
        filepath
            The path of the file to read
        jump_sequence
            The expected sequence of jumps in the data

        Returns
        -------
        The parsed data in the file
        """
        right = DataReader._read_csv(f"{filepath}_R.CSV", nb_headers_rows=4)
        left = DataReader._read_csv(f"{filepath}_L.CSV", nb_headers_rows=4)
        if sum(right.t - left.t) != 0.0:
            raise RuntimeError("Left and Right sensor data don't match")

        both = Data()
        both.t = right.t
        both.y = (np.sum(right.y, axis=1) + np.sum(left.y, axis=1))[:, np.newaxis]
        return ForceSensorData(both, jump_sequence=jump_sequence)

    @staticmethod
    def _read_csv(
        filepath, nb_sensors: int = None, nb_rows: int = None, nb_headers_rows: int = 0, conversion_factor: float = 1
    ) -> Data:
        """
        Read the actual file, assuming 'ncols' in the data

        Parameters
        ----------
        filepath
            The path for the file to read
        nb_sensors
            The number of data sensors to read. If not provided, it reads all of them, assuming the number of sensors
            is the number of columns in the first row data
        nb_rows
            The maximum number of rows to read, if 'None' it reads all
        nb_headers_rows
            The number of header rows (they are skipped)
        conversion_factor
            The factor to convert the data

        Returns
        -------
        The parsed data in the file
        """

        out = Data(nb_sensors=nb_sensors, conversion_factor=conversion_factor) if nb_sensors is not None else None

        with open(filepath) as csvfile:
            rows = csv.reader(csvfile, delimiter=",")
            for i, row in enumerate(rows):
                if i < nb_headers_rows:
                    continue
                if nb_rows is not None and i >= nb_rows + nb_headers_rows:
                    break

                if out is None:
                    nb_sensors = len(row[1:])
                    out = Data(nb_sensors=nb_sensors, conversion_factor=conversion_factor)

                out.append(float(row[0]), [float(i) for i in row[1 : nb_sensors + 1]])

        return out

    @staticmethod
    def fetch_trial_names(folder) -> tuple[str, ...]:
        """
        Fetch the unique names of all the file in the designated folder. The file name consists of numbers only

        Parameters
        ----------
        folder
            The folder to fetch the names from

        Returns
        -------
        A tuple of all the names
        """

        all_filenames = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        root_names = [re.split(r"^([0-9_.]*)_([a-zA-Z_.]*)(.CSV)$", filename)[1] for filename in all_filenames]
        unique_names = set(root_names)
        return tuple(sorted(unique_names))
