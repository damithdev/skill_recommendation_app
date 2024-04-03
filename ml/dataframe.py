import pandas as pd
from threading import Lock

import config


class DataFrameSingleton:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DataFrameSingleton, cls).__new__(cls)
                # Initialize the dataframe here
                cls._instance.dataframe = cls._load_dataframe()
        return cls._instance

    @staticmethod
    def _load_dataframe():
        # Load the dataframe here. This could be from a file, database, etc.
        # For example, loading from a CSV file:
        df = pd.read_csv(config.JOBS_FILE_PATH)
        return df

    def reload_dataframe(self):
        # Method to manually reload the dataframe
        self.dataframe = self._load_dataframe()

# Usage
# To access the dataframe:
# df_instance = DataFrameSingleton()
# dataframe = df_instance.dataframe
#
# # To reload the dataframe:
# df_instance.reload_dataframe()
