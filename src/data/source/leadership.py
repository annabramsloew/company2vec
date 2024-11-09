# # Adapted from: https://github.com/SocialComplexityLab/life2vec/blob/v1.0.0/src/data_new/sources/education.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import os

import dask.dataframe as dd
import pandas as pd

from ..decorators import save_parquet
from ..ops import sort_partitions
from ..logging_config import DATA_ROOT
from .base import FIELD_TYPE, TokenSource, Binned
from .source_helpers import active_participants_per_year


@dataclass
class LeadershipTokens(TokenSource):
    """This generates tokens based on information from the annual reports, registrations and currency datasets.
    Currently loads data from a CSV dump of the annual reports.

    :param input_csv: path to the Tables folder, from which data on registrations, currency rates and annual reports may be fetched.
    :param earliest_start: The earliest start date of a hospital encounter.
    """

    name: str = "leadership"
    fields: List[FIELD_TYPE] = field(
        default_factory=lambda: [
            "PARTICIPANT_TYPE", #Board member, CEO
            Binned("EXPERIENCE", prefix="EXPERIENCE", n_bins=20)
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
                PARTICIPANT_TYPE=lambda x: x.PARTICIPANT_TYPE.map({'BOARD_MEMBER': 'LEADER_BOARD', 'C_LEVEL_EXECUTIVE': 'LEADER_EXEC'},meta=('PARTICIPANT_TYPE', 'object')),
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
        result = self.parsed().dropna(subset='FromDate').set_index("CVR").rename(columns={'RelationType': 'PARTICIPANT_TYPE', 'FromDate': 'FROM_DATE', 'Experience': 'EXPERIENCE'})
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


        # interim data
        interim_path = DATA_ROOT / "interim" / "leadership"
        if os.path.exists(interim_path):


            files = os.listdir(interim_path)

            if len(files) > 0:
                
                # file paths ends with _YYYY.parquet. We want to get the last year
                years = [int(file.split("_")[-1].split(".")[0]) for file in files]
                latest_year = max(years)

                # Load the latest year
                ddf = dd.read_parquet(interim_path / f"active_participants_{latest_year}.parquet", dtype = {"FromDate": "timestamp[ns]",
                                                                                                            "CVR": int,
                                                                                                            "EntityID": int,
                                                                                                            "RelationType": str,
                                                                                                            "Experience": int})
                # ensure columns are correct types
                ddf['CVR'] = ddf['CVR'].astype(int)
                ddf['EntityID'] = ddf['EntityID'].astype(int)
                ddf['Experience'] = ddf['Experience'].astype(int)

                return ddf
            
            else:
                print("No interim data found at path: ", interim_path)
                pass
        else:
            raise ValueError("No interim data found at path: ", interim_path)

        columns_leadership = [
            "CVR",
            "EntityID",
            "ParticipantType",
            "RelationType",
            "Participation",
            "Date"
            ]

        output_columns = [
            "FROM_DATE",
            "CVR",
            "ENTITY_ID",
            "RELATION_TYPE",
            "EXPERIENCE"
            ]

        # Update the path to the data
        path_leadership = self.input_csv / "Participants"
        path_cvr = self.input_csv / "CVRFiltered"

        # Load files
        leadership_csv = [file for file in path_leadership.iterdir() if file.is_file() and file.suffix == '.csv']
        cvr_csv = [file for file in path_cvr.iterdir() if file.is_file() and file.suffix == '.csv']

        # Load data
        ddf_leadership = dd.read_csv(
            leadership_csv,
            usecols=columns_leadership,
            on_bad_lines="error",
            assume_missing=True,
            dtype={
                "CVR": int,
                "FromDate": str,
                "EntityID": int,
                "ParticipantType": str,
                "RelationType": str,
                "Participation": str,
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
        ddf_leadership = (ddf_leadership
            .loc[lambda x: x.RelationType != 'EJERANDEL']
            .loc[lambda x: x.ParticipantType != 'VIRKSOMHED'] #assuming we only want to keep persons as board members/c-level executives
        )

        #map management positions to PARTICIPANT_TYPE = ['BOARD_MEMBER', 'CEO']
        management_map = {
            "BESTYRELSESMEDLEM": "BOARD_MEMBER",
            "DIREKTION": "C_LEVEL_EXECUTIVE",
            "DIREKTØR": "C_LEVEL_EXECUTIVE",
            "FORMAND": "BOARD_MEMBER",
            "ADM. DIR.": "C_LEVEL_EXECUTIVE",
            "NÆSTFORMAND": "BOARD_MEMBER",
            "BESTYRELSE": "BOARD_MEMBER",
            "SUPPLEANT": "BOARD_MEMBER"
        }
        ddf_leadership['RelationType'] = ddf_leadership['RelationType'].str.strip().str.upper()
        ddf_leadership['RelationType'] = ddf_leadership['RelationType'].map(management_map)

        #filter away rows not in ParticipantType = ['BOARD_MEMBER', 'C_LEVEL_EXECUTIVE']
        ddf_leadership = ddf_leadership.loc[ddf_leadership['RelationType'].isin(['BOARD_MEMBER', 'C_LEVEL_EXECUTIVE'])]
        
        #turn date column into datetime and fill missing values with 2000-01-01
        ddf_leadership['Date'] = dd.to_datetime(ddf_leadership['Date'], errors='coerce')
        ddf_leadership['Date'] = ddf_leadership['Date'].fillna(pd.Timestamp('2000-01-01'))

        #filter away CVR's that are not in the lookup table from the employee data and registration data
        cvr_list = df_cvr['CVR'].compute()
        ddf_leadership = ddf_leadership.loc[ddf_leadership['CVR'].isin(cvr_list)]
        del df_cvr

        # Summarize leadership for every CVR per year (per Dec 31st)
        # List active ParticipantTypes and their corresponding experience
        ddf_leadership = active_participants_per_year(ddf_leadership)

        # Rename columns
        ddf = (ddf_leadership
            .rename(columns=dict(zip(ddf_leadership.columns, output_columns)))
        )

        if self.downsample:
            ddf = self.downsample_persons(ddf)

        assert isinstance(ddf, dd.DataFrame)

        return ddf
    

# Used for debugging
# if __name__ == "__main__":
#     report_tokens = LeadershipTokens()
#     report_tokens.tokenized()