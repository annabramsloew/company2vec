# # Adapted from: https://github.com/SocialComplexityLab/life2vec/blob/v1.0.0/src/data_new/sources/education.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import datetime

import dask.dataframe as dd
import pandas as pd

from ..decorators import save_parquet
from ..ops import sort_partitions
from ..logging_config import DATA_ROOT
from .base import FIELD_TYPE, TokenSource, Binned
from .source_helpers import dd_enrich_with_asof_values, bin_share

# ------------------------------------------ FIX IMPORTS ------------------------------------------
@dataclass
class OwnershipTokens(TokenSource):
    """This generates tokens based on information from the annual reports, registrations and currency datasets.
    Currently loads data from a CSV dump of the annual reports.

    :param input_csv: path to the Tables folder
    :param earliest_start: The earliest start date 
    """

    name: str = "ownership"
    fields: List[FIELD_TYPE] = field(
        default_factory=lambda: [
            "OWNER_TYPE", #internal or externals
            "SHARE", 
            "INDUSTRY",
            Binned("EMPLOYEE_COUNT", prefix="EMPLOYEES", n_bins=100) #TODO: THis bin should be the same as the one in the employee source
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
                OWNER_TYPE=lambda x: "OTYP_" + x.OWNER_TYPE.map({'EXTERNAL' : "EXT", "INTERNAL": "INT"}, meta=('OWNER_TYPE', 'object')), #OTYP : Ownership Type
                SHARE=lambda x: x.SHARE.apply(bin_share, meta=('SHARE', 'object')),
                INDUSTRY=lambda x: "IND_" + x.INDUSTRY.apply(lambda ind: ind[:4] if not pd.isna(ind) else "UNK", meta=('INDUSTRY', 'object'))
            ).pipe(sort_partitions, columns=["FROM_DATE"])[
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

        columns_owners = [
            "CVR",
            "EntityID",
            "ParticipantType",
            "RelationType",
            "Participation",
            "Date",
            "EquityPct"
            ]
        
        columns_registrations = ["CVR", "FromDate", "ChangeType", "NewValue"]
        
        columns_employee = ['CVR', 'FromDate', 'EmployeeCounts']

        output_columns = [
            "FROM_DATE",
            "CVR",
            "SHARE",
            "TYPE",
            "INDUSTRY",
            "EMPLOYEE_COUNT"
            ]

        # Update the path to the data
        path_owners = self.input_csv / "Participants"
        path_registrations = self.input_csv / "Registrations"
        path_employee = self.input_csv / "EmployeeCounts"
        path_cvr = self.input_csv / "CVRFiltered"

        # Load files
        owners_csv = [file for file in path_owners.iterdir() if file.is_file() and file.suffix == '.csv']
        registrations_csv = [file for file in path_registrations.iterdir() if file.is_file() and file.suffix == '.csv']
        employee_csv = [file for file in path_employee.iterdir() if file.is_file() and file.suffix == '.csv']
        cvr_csv = [file for file in path_cvr.iterdir() if file.is_file() and file.suffix == '.csv']

        # Load data
        ddf_owners = dd.read_csv(
            owners_csv,
            usecols=columns_owners,
            on_bad_lines="error",
            assume_missing=True,
            dtype={
                "CVR": int,
                "FromDate": str,
                "EntityID": int,
                "ParticipantType": str,
                "RelationType": str,
                "Participation": str,
                "EquityPct": float
            },
            blocksize="256MB",
        ).compute()

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
            blocksize="256MB",
        ).compute()

        ddf_employee = dd.read_csv(
            employee_csv,
            usecols=columns_employee,
            on_bad_lines="error",
            assume_missing=True,
            dtype={
                "CVR": int,
                "FromDate": str,
                "EmployeeCounts": int
            },
            blocksize="256MB",
        ).compute()

        df_cvr = dd.read_csv(
            cvr_csv,
            usecols=['CVR'],
            dtype={
                "CVR": int
            }
        )

        # Filter and enrich data
        #filter away ownertype person and non-owner participants and exit participations
        ddf_owners = (ddf_owners
            .loc[lambda x: x.RelationType == 'EJERANDEL']
            .loc[lambda x: x.ParticipantType == 'VIRKSOMHED']
            .loc[lambda x: x.Participation == 'entry']
        )
        
        #turn date column into datetime and fill missing values with 2000-01-01 and filter away dates before the earliest start date
        ddf_owners['Date'] = dd.to_datetime(ddf_owners['Date'], errors='coerce')
        ddf_owners['Date'] = ddf_owners['Date'].fillna(pd.Timestamp('2000-01-01'))
        ddf_owners = ddf_owners.loc[lambda x: x.Date >= self.earliest_start]

        #filter away CVR's that are not in the lookup table from the employee data and registration data
        cvr_list = df_cvr['CVR'].compute()
        ddf_owners = ddf_owners[ddf_owners['CVR'].isin(cvr_list)]
        ddf_registrations = ddf_registrations[ddf_registrations['CVR'].isin(cvr_list)]
        ddf_employee = ddf_employee[ddf_employee['CVR'].isin(cvr_list)]
        del df_cvr

        #filter away employee data with dates before the earliest start date
        ddf_employee['FromDate'] = dd.to_datetime(ddf_employee['FromDate'], errors='coerce')
        ddf_employee = ddf_employee.loc[lambda x: x.FromDate >= self.earliest_start].rename(columns={'FromDate': 'Date'})

        #compute owner type "internal" - rename EntityID to JoinID
        ddf_owners_internal = (ddf_owners
            .assign(OwnerType = 'INTERNAL')
            .drop(columns=['ParticipantType', 'RelationType', 'Participation'])
            .rename(columns={'EntityID': 'JoinID'})
        )

        #compute owner type "external" - rename CVR to JoinID and EntityID to CVR
        ddf_owners_external = (ddf_owners
            .assign(OwnerType = 'EXTERNAL')
            .drop(columns=['ParticipantType', 'RelationType', 'Participation'])
            .rename(columns={'CVR': 'JoinID', 'EntityID': 'CVR'})
        )

        #rename CVR column in Registrations and Employees to CVR_right for the asof merge
        ddf_registrations = ddf_registrations.rename(columns={'CVR': 'CVR_right'})
        ddf_employee = ddf_employee.rename(columns={'CVR': 'CVR_right'})

        # enrich owners with asof values from the registrations - industry
        ddf_owners_internal = dd_enrich_with_asof_values(
            ddf_owners_internal, 
            ddf_registrations, 
            values=['Industry'], 
            date_col_df='Date', 
            date_col_registrations='FromDate',
            left_by_value='JoinID',
            right_by_value='CVR_right'
        ).drop(columns=['CVR_right'])
        
        ddf_owners_external = dd_enrich_with_asof_values(
            ddf_owners_external,
            ddf_registrations,
            values=['Industry'],
            date_col_df='Date',
            date_col_registrations='FromDate',
            left_by_value='JoinID',
            right_by_value='CVR_right'
        ).drop(columns=['CVR_right'])
        del ddf_registrations
        
        # sort the dataframes by the date column
        ddf_owners_internal = ddf_owners_internal.set_index('Date', drop=False).reset_index(drop=True)
        ddf_owners_internal = ddf_owners_internal.sort_values('Date')
        ddf_owners_external = ddf_owners_external.set_index('Date', drop=False).reset_index(drop=True)
        ddf_owners_external = ddf_owners_external.sort_values('Date')
        ddf_employee = ddf_employee.set_index('Date', drop=False).reset_index(drop=True)
        ddf_employee = ddf_employee.sort_values('Date')

        # merge employee data with owners data with asof merge
        ddf_owners_internal = dd.merge_asof(
            ddf_owners_internal,
            ddf_employee,
            on='Date',
            left_by='JoinID',
            right_by='CVR_right',
            direction='backward'
        ).drop(columns=['JoinID','CVR_right'])

        ddf_owners_external = dd.merge_asof(
            ddf_owners_external,
            ddf_employee,
            on='Date',
            left_by= 'JoinID',
            right_by='CVR_right',
            direction='backward'
        ).drop(columns=['JoinID', 'CVR_right'])
        del ddf_employee

        #concatenate the internal and external owners
        ddf_owners = dd.concat([dd.from_pandas(ddf_owners_internal, npartitions=1), dd.from_pandas(ddf_owners_external, npartitions=1)])


        column_map = {
            "Date":"FROM_DATE",
            "Industry":"INDUSTRY",
            "OwnerType":"OWNER_TYPE",
            "EmployeeCounts":"EMPLOYEE_COUNT",
            "EquityPct":"SHARE"
        }

        # Rename columns
        ddf = (ddf_owners
            .rename(columns=column_map)
        )

        ddf = ddf.fillna({
            'INDUSTRY': 'UNK',
            'EMPLOYEE_COUNT': 0
        }).dropna(subset=['SHARE'])

        
        #check result
        #ddf = ddf.compute()

        if self.downsample:
            ddf = self.downsample_persons(ddf)

        assert isinstance(ddf, dd.DataFrame)

        return ddf
    


# #Used for debugging
# if __name__ == "__main__":
#     report_tokens = OwnershipTokens()
#     report_tokens.tokenized()