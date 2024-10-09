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
from .source_helpers import dd_enrich_with_asof_values, convert_currency


@dataclass
class ProductionUnitTokens(TokenSource):
    """This generates tokens based on information from the ProductionUnits table.

    :param input_csv: Path to folder where the ProductionUnits table is stored.
    :param earliest_start: TODO?
    """

    name: str = "production_units"
    fields: List[FIELD_TYPE] = field(
        default_factory=lambda: [
            "ACTION", #open or close
            "INDUSTRY", 
            "MUNICIPALITY"
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
            .assign(
                ACTION=lambda x: x.ACTION.map({"entry": "ACT_OPEN", "exit": "ACT_CLOSE"}, meta=('ACTION', 'object')),
                INDUSTRY=lambda x: "IND_" + x.INDUSTRY.apply(lambda ind: ind[:4] if not ind=="UNK" else "UNK", meta=('INDUSTRY', 'object')), 
                MUNICIPALITY=lambda x: "WMUN_" + x.MUNICIPALITY
            )
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

        columns_punits = [
            "CVR",
            "Date",
            "ChangeType",
            "Industry",
            "Municipality"
            ]

        output_columns = [
            "CVR",
            "FROM_DATE",
            "ACTION",
            "INDUSTRY",
            "MUNICIPALITY"
            ]

        # Update the path to the data
        path_punits = self.input_csv / "ProductionUnits"
        path_cvr = self.input_csv / "CVRFiltered"

        # Load files
        punits_csv = [file for file in path_punits.iterdir() if file.is_file() and file.suffix == '.csv']
        cvr_csv = [file for file in path_cvr.iterdir() if file.is_file() and file.suffix == '.csv']

        # Load data
        ddf_punits = dd.read_csv(
            punits_csv,
            usecols=columns_punits,
            on_bad_lines="error",
            assume_missing=True,
            dtype={
                "CVR": int,
                "Date": str,
                "ChangeType": str,
                "Industry": str,
                "Municipality": str
            },
            blocksize="256MB",
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
        ddf_punits = ddf_punits.loc[ddf_punits['CVR'].isin(cvr_list)]

        # Filter out data and rename columns
        ddf = (ddf_punits
            .rename(columns=dict(zip(ddf_punits.columns, output_columns)))
            .loc[lambda x: x.FROM_DATE >= self.earliest_start]
        )

        # Handle missing values and deal with datatypes
        ddf = ddf.fillna({
            'COMPANY_STATUS': 'UNK',
            'INDUSTRY': 'UNK',
            'MUNICIPALITY': 'UNK',
        })

        if self.downsample:
            ddf = self.downsample_persons(ddf)

        assert isinstance(ddf, dd.DataFrame)

        return ddf
    
#use for debugging
if __name__ == "__main__":
    tokens = ProductionUnitTokens()
    parsed_data = tokens.tokenized().compute()