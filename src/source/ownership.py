# # Adapted from: https://github.com/SocialComplexityLab/life2vec/blob/v1.0.0/src/data_new/sources/education.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import datetime

import dask.dataframe as dd
import pandas as pd

#from ..decorators import save_parquet
from ..ops import sort_partitions
#from ..serialize import DATA_ROOT
from .base import FIELD_TYPE, TokenSource, Binned
from .source_helpers import dd_enrich_with_asof_values, convert_currency

DATA_ROOT = Path.home() / "Library" / "CloudStorage" / "OneDrive-DanmarksTekniskeUniversitet(2)" / "Virk2Vec" / "data"

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
            "TYPE", #internal or external
            "SHARE", #TODO should this be a binned field?
            "INDUSTRY",
            Binned("EMPLOYEE_COUNT", prefix="EMPLOYEES", n_bins=100)
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
                "ParticipantType": str,
                "RelationType": str,
                "Participation": str,
                "EquityPct": float
            },
            blocksize="256MB",
        )

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
        )

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
        )

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

        #convert EntityID to int64
        ddf_owners['EntityID'] = ddf_owners['EntityID'].astype('int64')

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
        ddf_owners = dd.concat([ddf_owners_internal, ddf_owners_external])

        # Rename columns
        ddf = (ddf_owners
            .rename(columns=dict(zip(ddf_owners.columns, output_columns)))
        )

        #check min and max date
        #min_date = ddf['FROM_DATE'].min().compute()
        #max_date = ddf['FROM_DATE'].max().compute()

        #check result
        #ddf = ddf.compute()

        if self.downsample:
            ddf = self.downsample_persons(ddf)

        assert isinstance(ddf, dd.DataFrame)

        return ddf
    


# Used for debugging
# if __name__ == "__main__":
#     report_tokens = AnnualReportTokens()
#     report_tokens.parsed()