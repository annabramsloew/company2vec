# # Adapted from: https://github.com/SocialComplexityLab/life2vec/blob/v1.0.0/src/data_new/sources/education.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import dask.dataframe as dd
import pandas as pd

from ..decorators import save_parquet
from ..ops import sort_partitions
from ..serialize import DATA_ROOT
from .base import FIELD_TYPE, TokenSource, Binned




@dataclass
class CapitalTokens(TokenSource):
    """This generates tokens based on information from the annual reports, registrations and currency datasets.
    Currently loads data from a CSV dump of the annual reports.

    :param input_csv: path to the Tables folder, from which data on registrations, currency rates and annual reports may be fetched.
    :param earliest_start: The earliest start date of a hospital encounter.
    """

    name: str = "capital"
    fields: List[FIELD_TYPE] = field(
        default_factory=lambda: [
            "PAYMENT_TYPE", #cash, converting debt, transferring profit, etc.
            Binned("RATE", prefix="RATE", n_bins=10),
            Binned("INVESTMENT", prefix="INV", n_bins=10),
        ]
    )

    input_csv: Path = DATA_ROOT / "Tables"
    earliest_start: str = "01/01/2013"

    def _post_init__(self) -> None:
        self._earliest_start = pd.to_datetime(self.earliest_start)

    
    @save_parquet(
        DATA_ROOT / "processed/sources/{self.name}/tokenized",
        on_validation_error="error",
    )

    def tokenized(self) -> dd.DataFrame:
        """
        Loads the indexed data, then tokenizes it.
        """

        result = (
            self.indexed()
            .assign( #note to self: https://erhvervsstyrelsen.dk/vejledning-anvendelse-af-vurderingsberetninger-ved-registreringer#chapter4-3
                PAYMENT_TYPE=lambda x: x.PAYMENT_TYPE.map({
                                                            "kontant": "PAY_CASH", 
                                                            "ved overførte reserver / overskud": "PAY_PROFIT",
                                                            "ved konvertering af gæld": "PAY_DEBT",
                                                            "i værdier": "PAY_ASSETS",
                                                            "ved fusion" : "PAY_MERGER",
                                                            "ved indskud af bestemmende kapitalpost" : "PAY_MERGER",
                                                            "ved indskud af bestående virksomhed": "PAY_MERGER",
                                                            "ved ombytning af konvertible gældsbreve": "PAY_DEBT",
                                                            '[UNK]': '[UNK]'
                                                            }, meta=('PAYMENT_TYPE', 'object')
                                                           )
            )
            .pipe(sort_partitions, columns=["FROM_DATE"])[
                ["FROM_DATE", *self.field_labels()]
            ]
        )
        assert isinstance(result, dd.DataFrame)
        return result


    @save_parquet(
        DATA_ROOT / "interim/sources/{self.name}/indexed",
        on_validation_error="recompute",
    )


    def indexed(self) -> dd.DataFrame:
        """Loads the parsed data, sets the index, then saves the indexed data"""
        result = self.parsed().set_index("CVR")
        assert isinstance(result, dd.DataFrame)
        return result


    @save_parquet(
        DATA_ROOT / "interim/sources/{self.name}/parsed",
        on_validation_error="error",
        verify_index=False,
    )


    def parsed(self) -> dd.DataFrame:
        """Parses the CSV file, applies some basic filtering, then saves the result
        as compressed parquet file, as this is easier to parse than the CSV for the
        next steps"""

        columns_capital = [
            "CVR",
            "Date",
            "InvestmentDKK",
            "Rate",
            "PaymentType",
            "InvestmentType"
            ]

        output_columns = [
            "CVR",
            "FROM_DATE",
            "PAYMENT_TYPE",
            "RATE",
            "INVESTMENT"
            ]

        # Update the path to the data
        path_capitalchanges = self.input_csv / "CapitalChanges"
        path_cvr = self.input_csv / "CVRFiltered"

        # Load files
        capital_csv = [file for file in path_capitalchanges.iterdir() if file.is_file() and file.suffix == '.csv']
        cvr_csv = [file for file in path_cvr.iterdir() if file.is_file() and file.suffix == '.csv']

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
                "PaymentType": str,
                "InvestmentType": str
            },
            blocksize="256MB",
            lineterminator='\n'
        )

        df_cvr = dd.read_csv(
            cvr_csv,
            usecols=['CVR'],
            dtype={
                "CVR": int
            }
        )

        # filter away CVR's that are not in the lookup table from the annual report data and registration data
        cvr_list = df_cvr['CVR'].compute()
        ddf_capital = ddf_capital.loc[ddf_capital['CVR'].isin(cvr_list)]
        
        # Handle data types and compute total investment, multiply by -1 if 'InvestmentType' is 'decrease'   
        ddf = (ddf_capital
            .loc[ddf_capital["InvestmentType"] == 'Kapitalforhøjelse']
            .assign(Date=lambda df: dd.to_datetime(df["Date"], errors='coerce', format='%d-%m-%Y'))
            .assign(Investment=lambda df: df["InvestmentDKK"] * (df["Rate"] / 100))
            .drop(columns=['InvestmentDKK', 'InvestmentType'])
        )
        column_map = {
            "CVR": "CVR",
            "Date": "FROM_DATE",
            "InvestmentDKK" : "INVESTMENT",
            "Rate" : "RATE",
            "PaymentType" : "PAYMENT_TYPE"
        }
        # Filter out data and rename columns
        ddf = (ddf
            .rename(columns=column_map)
            .loc[lambda x: x.FROM_DATE >= self.earliest_start]
        )

        ddf = ddf.fillna({'PAYMENT_TYPE': "[UNK]"})

        if self.downsample:
            ddf = self.downsample_persons(ddf)

        assert isinstance(ddf, dd.DataFrame)

        return ddf
    
#use for debugging
# if __name__ == "__main__":
#     tokens = CapitalTokens()
#     parsed_data = tokens.tokenized().compute()