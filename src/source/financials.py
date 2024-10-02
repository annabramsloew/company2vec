# # Adapted from: https://github.com/SocialComplexityLab/life2vec/blob/v1.0.0/src/data_new/sources/education.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import dask.dataframe as dd
import pandas as pd

#from ..decorators import save_parquet
from ..ops import sort_partitions
#from ..serialize import DATA_ROOT
from .base import FIELD_TYPE, TokenSource, Binned
from .source_helpers import dd_enrich_with_asof_values, convert_currency


DATA_ROOT = Path.home() / "Library" / "CloudStorage" / "Dropbox" / "DTU" / "Virk2Vec"

# ------------------------------------------ FIX IMPORTS ------------------------------------------
@dataclass
class AnnualReportTokens(TokenSource):
    """This generates tokens based on information from the annual reports, registrations and currency datasets.
    Currently loads data from a CSV dump of the annual reports.

    :param input_csv: path to the Tables folder, from which data on registrations, currency rates and annual reports may be fetched.
    :param earliest_start: The earliest start date of a hospital encounter.
    """

    name: str = "financaial"
    fields: List[FIELD_TYPE] = field(
        default_factory=lambda: [
            "COMPANY_TYPE", 
            "INDUSTRY", 
            "COMPANY_STATUS", 
            "ADDRESS", #TODO: Change
            Binned("PROFIT_LOSS", prefix="PROFIT_LOSS", n_bins=100),
            Binned("EQUITY", prefix="EQUITY", n_bins=100),
            Binned("ASSETS", prefix="ASSETS", n_bins=100),
            Binned("LIABILITIES_AND_EQUITY", prefix="LIABILITIES_AND_EQUITY", n_bins=100)
        ]
    )

    input_csv: Path =  Path(r"/Users/nikolaibeckjensen/Dropbox/Virk2Vec/Tables") #TODO: Change back to relative import DATA_ROOT / "Tables"
    earliest_start: str = "01/01/2013"

    def _post_init__(self) -> None:
        self._earliest_start = pd.to_datetime(self.earliest_start)

    """
    @save_parquet(
        DATA_ROOT / "processed/sources/{self.name}/tokenized",
        on_validation_error="error",
    )
    """

    # ------------------------------------------ TODO FIX TOKENIZED ------------------------------------------
    def tokenized(self) -> dd.DataFrame:
        """
        Loads the indexed data, then tokenizes it.
        """

        result = (
            self.indexed()
            .assign(
                COMPANY_TYPE=lambda x: "CTYP_" + x.COMPANY_TYPE.map({"A/S": "AS", 
                                                                     "ApS": "APS", 
                                                                     "IVS": "IVS"}),
                INDUSTRY=lambda x: "IND_" + x.INDUSTRY, 
                COMPANY_STATUS=lambda x: "CSTAT_" + x.COMPANY_STATUS, #TODO: Define status mapping
                ADDRESS=lambda x: "WMUN_" + x.ADDRESS, #TODO: Change
            )
            .pipe(sort_partitions, columns=["FROM_DATE"])[["FROM_DATE", *self.field_labels()]]
        )

        assert isinstance(result, dd.DataFrame)
        return result


    """
    @save_parquet(
        DATA_ROOT / "interim/sources/{self.name}/indexed",
        on_validation_error="recompute",
    )
    """
    # ------------------------------------------ TODO FIX TOKENIZED ------------------------------------------

    def indexed(self) -> dd.DataFrame:
        """Loads the parsed data, sets the index, then saves the indexed data"""
        result = self.parsed().set_index("CVR")
        assert isinstance(result, dd.DataFrame)
        return result

    """
    @save_parquet(
        DATA_ROOT / "interim/sources/{self.name}/parsed",
        on_validation_error="error",
        verify_index=False,
    )
    """

    def parsed(self) -> dd.DataFrame:
        """Parses the CSV file, applies some basic filtering, then saves the result
        as compressed parquet file, as this is easier to parse than the CSV for the
        next steps"""

        columns_registrations = [
        "CVR",
        "FromDate",
        "ChangeType",
        "NewValue"
        ]

        columns_annualreport = [
            "CVR",
            "PublicationDate",
            "Currency",
            "ProfitLoss",
            "Equity", 
            "Assets", 
            "LiabilitiesAndEquity"
        ]

        columns_currency = [
        "year",
        "month",
        "from_currency",
        "rate"
        ]

        output_columns = [
        "START_DATE",
        "CVR",
        "PROFIT_LOSS",
        "ASSETS",
        "EQUITY",
        "LIABILITIES_AND_EQUITY",
        "INDUSTRY",
        "COMPANY_TYPE",
        "MUNICIPALITY",
        "COMPANY_STATUS"
        ]
        
        # Update the path to the data
        path_financials = self.input_csv / "Financials"
        path_registrations = self.input_csv  / "Registrations"
        path_currency = self.input_csv / "Currency"
        
        # Load files
        financials_csv = [file for file in path_financials.iterdir() if file.is_file() and file.suffix == '.csv']
        financials_csv = [financials_csv[0]]
        registrations_csv = [file for file in path_registrations.iterdir() if file.is_file() and file.suffix == '.csv'] 
        currency_csv = [file for file in path_currency.iterdir() if file.is_file() and file.suffix == '.csv']

        # Load data
        ddf_registrations = dd.read_csv(
            registrations_csv,
            usecols=columns_registrations,
            on_bad_lines="error",
            assume_missing=True,
            dtype={
                "CVR": int,
                "FromDate": str,
                "ChangeType": str,
                "NewValue": str
            },
            blocksize="256MB"
        )
        
        ddf_annualreport = dd.read_csv(
            financials_csv,
            usecols=columns_annualreport,
            on_bad_lines="error",
            assume_missing=True,
            dtype={
                "CVR": int,
                "PublicationDate": str,
                "Currency": str,
                "ProfitLoss": float,
                "Equity": float,
                "Assets": float,
                "LiabilitiesAndEquity": float,
            },
            blocksize="256MB"
        )
        
        ddf_currency = dd.read_csv(
            currency_csv,
            usecols=columns_currency,
            on_bad_lines="error",
            assume_missing=True,
            dtype={
                "year": int,
                "month": int,
                "from_currency": str,
                "rate": float
            },
            blocksize="256MB"
        )
        
        # enrich the annual report with asof values from the registrations
        ddf = dd_enrich_with_asof_values(
            ddf_annualreport, 
            ddf_registrations, 
            values=['Industry', 'CompanyType', 'Address', 'Status'], 
            date_col_df='PublicationDate', 
            date_col_registrations='FromDate'
            )
        
        # convert currency
        ddf = convert_currency(ddf, ddf_currency, 
                        amount_cols=['ProfitLoss', 'Equity', 'Assets', 'LiabilitiesAndEquity'], 
                        currency_col='Currency', 
                        date_col='PublicationDate')
        
        # Drop 'Currency' column, rename columns, and drop values
        ddf = (
            ddf.drop(columns=['Currency'])
            .pipe(lambda df: df.rename(columns=dict(zip(df.columns, output_columns))))
            .loc[lambda x: x.START_DATE >= self.earliest_start]
        )

        if self.downsample:
            ddf = self.downsample_persons(ddf)

        assert isinstance(ddf, dd.DataFrame)

        return ddf
    

# use for debugging
if __name__ == "__main__":
    tokens = AnnualReportTokens()
    parsed_data = tokens.tokenized().compute()
    