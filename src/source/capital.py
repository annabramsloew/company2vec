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

# TODO: MODIFY TO INSTEAD USE FINANCIAL DATA

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
            "PAYMENT_TYPE", #cash, converting debt, transferring profit, etc.
            #"INVESTMENT_TYPE", #increase or decrease 
            Binned("RATE", prefix="RATE", n_bins=100),
            Binned("INVESTMENT", prefix="INVESTMENT", n_bins=100),
        ]
    )

    input_csv: Path = DATA_ROOT / "Tables"
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
        Clamps the C_ADIAG field, and converts C_INDM and C_PATTYPE to strings.
        """

        result = (
            self.indexed()
            .assign(
                C_ADIAG=lambda x: x.C_ADIAG.str[1:4],
                C_INDM=lambda x: x.C_INDM.map(
                    {"1": "URGENT", "2": "NON_URGENT"}
                ).astype("string"),
                C_PATTYPE=lambda x: x.C_PATTYPE.map(
                    {"0": "INPAT", "2": "OUTPAT", "3": "EMERGENCY"}
                ).astype("string"),
            )
            .pipe(sort_partitions, columns=["START_DATE"])[
                ["START_DATE", *self.field_labels()]
            ]
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

        columns_capital = [
            "CVR",
            "Date",
            "InvestmentDKK",
            "Rate",
            "PaymentType"
            #InvestmentType
            ]

        output_columns = [
            "CVR",
            "START_DATE",
            "PAYMENT_TYPE",
            "RATE",
            "INVESTMENT"
            ]

        # Update the path to the data
        path_capitalchanges = self.input_csv / "CapitalChanges"

        # Load files
        capital_csv = [file for file in path_capitalchanges.iterdir() if file.is_file() and file.suffix == '.csv']

        # Load data
        ddf_capital = dd.read_csv(
            capital_csv,
            usecols=columns_capital,
            on_bad_lines="error",
            assume_missing=True,
            dtype={
                "CVR": int,
                "Date": str,
                "InvestmentDKK": float,
                "Rate": float,
                "PaymentType": str #,
                #"InvestmentType": str
            },
            blocksize="256MB",
            lineterminator='\n'
        )
        
        # Handle data types and compute total investment, multiply by -1 if 'InvestmentType' is 'decrease'   
        ddf = (ddf_capital
            .assign(Date=lambda df: dd.to_datetime(df["Date"], errors='coerce', format='%d-%m-%Y'))
            .assign(Investment=lambda df: df["InvestmentDKK"] * (df["Rate"] / 100))
            #.assign(Investment=lambda df: df["Investment"] * df["InvestmentType"].map({'decrease': -1}).fillna(1))
            .drop(columns=['InvestmentDKK'])#,'InvestmentType'])
        )

        # Filter out data and rename columns
        ddf = (ddf
            .rename(columns=dict(zip(ddf.columns, output_columns)))
            .loc[lambda x: x.START_DATE >= self.earliest_start]
        )

        if self.downsample:
            ddf = self.downsample_persons(ddf)

        assert isinstance(ddf, dd.DataFrame)

        return ddf