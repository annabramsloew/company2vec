
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

from datetime import datetime
import dask.dataframe as dd
import pandas as pd
import os

from ..decorators import save_parquet
from ..ops import sort_partitions
from ..serialize import DATA_ROOT
from .base import FIELD_TYPE, TokenSource, Binned

from .source_helpers import enrich_with_asof_values

@dataclass
class StatusTokens(TokenSource):
    """This generates tokens based on information from the Employee and Registrations table .

    :param input_csv: Path to folder where the Employee and Registrations tables are stored.
    :param earliest_start: The earliest start date of a hospital encounter.
    """

    name: str = "status"
    fields: List[FIELD_TYPE] = field(
        default_factory=lambda: [
            "COMPANY_STATUS", 
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
                COMPANY_STATUS=lambda x: x.COMPANY_STATUS.map({
                                                                "NORMAL": "CSTAT_ACTIVE",
                                                                "AKTIV": "CSTAT_ACTIVE",
                                                                "UNDER FRIVILLIG LIKVIDATION" : "CSTAT_ONGOING_LIQUIDATION",
                                                                "UNDER TVANGSOPLØSNING" : "CSTAT_ONGOING_DISSOLUTION",
                                                                "UNDER KONKURS" : "CSTAT_ONGOING_BANKRUPTCY",
                                                                "UNDER REKONSTRUKTION" : "CSTAT_ONGOING_RECONSTRUCTION",
                                                                "UNDER REASSUMERING" : "CSTAT_ONGOING_REASSUMPTION",
                                                                "OPLØST EFTER ERKLÆRING" : "CSTAT_DISSOLVED_DECLARATION",
                                                                "OPLØST EFTER GRÆNSEOVERSKRIDENDE FUSION" : "CSTAT_DISSOLVED_MERGER",
                                                                "OPLØST EFTER SPALTNING" : "CSTAT_DISSOLVED_MERGER",
                                                                "OPLØST EFTER FUSION" : "CSTAT_DISSOLVED_MERGER",
                                                                "OPLØST EFTER FRIVILLIG LIKVIDATION" : "CSTAT_DISSOLVED_LIQUIDATION",
                                                                'OPLØST EFTER GRÆNSEOVERSKRIDENDE HJEMSTEDSFLYTNING' : "CSTAT_DISSOLVED_MIGRATION",
                                                                "SLETTET" : "CSTAT_DISSOLVED",
                                                                "OPLØST EFTER KONKURS" : "CSTAT_DISSOLVED",
                                                                "TVANGSOPLØST" : "CSTAT_DISSOLVED",
                                                                "UDEN RETSVIRKNING" : "CSTAT_NO_LEGAL_EFFECT",
                                                                "[UNK]" : "[UNK]"
                                                            }, meta=('COMPANY_STATUS', 'object'))
            )
            .pipe(sort_partitions, columns=["FROM_DATE"])[["FROM_DATE", *self.field_labels()]]
        )

        assert isinstance(result, dd.DataFrame)
        return result



    @save_parquet(
        DATA_ROOT / "interim/sources/{self.name}/indexed",
        on_validation_error="recompute",
    )
    def indexed(self) -> dd.DataFrame:
        """Loads the parsed data, sets the index, then saves the indexed data"""
        result = self.parsed().dropna(subset='FROM_DATE').set_index("CVR")
        assert isinstance(result, dd.DataFrame)
        return result



    def parsed(self) -> dd.DataFrame:
        """Parses the CSV file, applies some basic filtering, then saves the result
        as compressed parquet file, as this is easier to parse than the CSV for the
        next steps"""

        # read CVR lookup table to use for filtering
        df_cvr = pd.read_csv(self.input_csv / "CVRFiltered" / "CVR_list.csv", index_col=0)
        
        path_registrations = self.input_csv / "Registrations"
        registrations_csv = [file for file in path_registrations.iterdir() if file.is_file() and file.suffix == '.csv']

        # Load data
        df_registrations = dd.read_csv(
            registrations_csv,
            usecols=["CVR", "FromDate", "ChangeType", "NewValue"],
            on_bad_lines="error",
            assume_missing=True,
            dtype={
                "CVR": int,
                "FromDate": str,
                "ChangeType": str,
                "NewValue": str
            },
            blocksize="256MB",
        ).compute()


        # choose only the relevant cvr numbers
        df_registrations = df_registrations.loc[df_registrations['CVR'].isin(df_cvr['CVR'].values)]

        # limit registrations to the earliest_start
        df_registrations['FromDate'] = pd.to_datetime(df_registrations['FromDate'], errors='coerce')
        df_registrations.dropna(subset=['FromDate'], inplace=True)
        df_registrations = df_registrations.loc[df_registrations['FromDate'] >= self.earliest_start]
        
        # choose only the relevant change types
        df_registrations = df_registrations.loc[df_registrations['ChangeType'] == 'Status']
        df_registrations = df_registrations[['CVR', 'FromDate', 'NewValue']].rename(columns={'FromDate':'FROM_DATE', 'NewValue': 'COMPANY_STATUS'})
        
        # format company status
        df_registrations['COMPANY_STATUS'] = df_registrations['COMPANY_STATUS'].str.upper()
        df_registrations = df_registrations.fillna({'COMPANY_STATUS': '[UNK]'})

        # convert back to dask dataframe
        ddf = dd.from_pandas(df_registrations, npartitions=5)
        
        assert isinstance(ddf, dd.DataFrame)

        return ddf
    

#use for debugging
if __name__ == "__main__":
    tokens = StatusTokens()
    parsed_data = tokens.parsed()
    parsed_data = tokens.tokenized()
    a = 1