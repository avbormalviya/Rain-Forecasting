from pathlib import Path
from zipfile import ZipFile
from abc import ABC, abstractmethod

import pandas as pd


# define constants
BASE_DIR = Path(__file__).resolve().parent.parent


# define abstract class for data ingestion
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """ Abstract method for data ingestion from given file path """
        ...


class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """ Ingest data from zip file """
        # if file path is not a zip file, raise error
        if not Path(file_path).suffix == ".zip":
            raise ValueError("File path must be a zip file")

        # Extract the zip file
        with ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR / "extracted_data")

        # Find CSV files in the extracted data
        extracted_files = Path(BASE_DIR / "extracted_data")
        csv_files = list(extracted_files.rglob("*.csv"))

        for file in csv_files:
            print(file)

        # Check if there is only zero or multiple CSV files in the extracted data
        if len(csv_files) == 0:
            raise ValueError("No CSV files found in the extracted data")
        if len(csv_files) > 1:
            raise ValueError("Multiple CSV files found in extracted data")

        # read the CSV file as DataFrame
        df = pd.read_csv(csv_files[0])

        # return the DataFrame
        return df


# Implement the factory to create DataIngestors
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """ Return the appropriate DataIngestor Base on file extension """
        if file_extension == '.zip':
            return ZipDataIngestor()
        else:
            raise ValueError("Unsupported file extension")


# Example usage:
if __name__ == "__main__":
    file_path = BASE_DIR / "data" / "archive.zip"
    file_extension = Path(file_path).suffix
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    df = data_ingestor.ingest(file_path)
    print(df.head())