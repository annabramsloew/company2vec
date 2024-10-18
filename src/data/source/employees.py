# # Adapted from: https://github.com/SocialComplexityLab/life2vec/blob/v1.0.0/src/data_new/sources/education.py

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
class EmployeeTokens(TokenSource):
    """This generates tokens based on information from the Employee and Registrations table .

    :param input_csv: Path to folder where the Employee and Registrations tables are stored.
    :param earliest_start: The earliest start date of a hospital encounter.
    """

    name: str = "employees"
    fields: List[FIELD_TYPE] = field(
        default_factory=lambda: [
            "COMPANY_TYPE", 
            "INDUSTRY", 
            "COMPANY_STATUS", 
            "MUNICIPALITY", 
            Binned("EMPLOYEE_COUNT", prefix="EMPLOYEES", n_bins=100)
        ]
    )
    
    input_csv: Path = Path(r"/Users/nikolaibeckjensen/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Virk2Vec/data/Tables")
    earliest_start: str = "01/01/2008"

    
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
                COMPANY_TYPE=lambda x: x.COMPANY_TYPE.map({"A/S": "CTYPE_AS", 
                                                            "APS": "CTYPE_APS", 
                                                            "IVS": "CTYPE_IVS",
                                                            "[UNK]" : "[UNK]"}, meta=('COMPANY_TYPE', 'object')),
                INDUSTRY=lambda x: x.INDUSTRY.apply(lambda ind: "IND_" + ind[:4] if ind != "[UNK]" else ind, meta=('INDUSTRY', 'object')), 
                COMPANY_STATUS=lambda x: x.COMPANY_STATUS.map({
                                                                "NORMAL": "CSTAT_ACTIVE",
                                                                "AKTIV": "CSTAT_ACTIVE",
                                                                "UNDER REASSUMERING" : "CSTAT_OTHER",
                                                                "UNDER FRIVILLIG LIKVIDATION" : "CSTAT_OTHER",
                                                                "UNDER REKONSTRUKTION" : "CSTAT_DISTRESS",
                                                                "UNDER TVANGSOPLØSNING" : "CSTAT_DISTRESS",
                                                                "UNDER KONKURS" : "CSTAT_DISTRESS",
                                                                "OPLØST EFTER GRÆNSEOVERSKRIDENDE FUSION" : "CSTAT_DISSOLVED",
                                                                "OPLØST EFTER GRÆNSEOVERSKRIDENDE HJEMSTEDSFLYTNING" : "CSTAT_DISSOLVED",
                                                                "OPLØST EFTER SPALTNING" : "CSTAT_DISSOLVED",
                                                                "OPLØST EFTER ERKLÆRING" : "CSTAT_DISSOLVED",
                                                                "OPLØST EFTER FUSION" : "CSTAT_DISSOLVED",
                                                                "OPLØST EFTER FRIVILLIG LIKVIDATION" : "CSTAT_DISSOLVED",
                                                                "SLETTET" : "CSTAT_DISSOLVED",
                                                                "OPLØST EFTER KONKURS" : "CSTAT_BANKRUPT",
                                                                "TVANGSOPLØST" : "CSTAT_BANKRUPT",
                                                                "[UNK]": "[UNK]"
                                                            }, meta=('COMPANY_STATUS', 'object')), 
                MUNICIPALITY=lambda x: x.MUNICIPALITY.apply(lambda mun: "MUN_" + mun if mun != "[UNK]" else mun, meta=('MUNICIPALITY', 'object'))
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
        result = self.parsed().set_index("CVR")
        assert isinstance(result, dd.DataFrame)
        return result



    def parsed(self) -> dd.DataFrame:
        """Parses the CSV file, applies some basic filtering, then saves the result
        as compressed parquet file, as this is easier to parse than the CSV for the
        next steps"""

        # read CVR lookup table to use for filtering
        df_cvr = pd.read_csv(self.input_csv / "CVRFiltered" / "CVR_list.csv", index_col=0)
        
        ddf_list = []
        files = os.listdir(self.input_csv / "EmployeeCounts")
        # discard files which are not csv
        files = [file for file in files if file.endswith('.csv')]
        
        for i in range(len(files)):
            print(i)

            # read chunk of employee data
            df_employees = pd.read_csv(self.input_csv / f"EmployeeCounts/chunk{i}.csv", index_col=0)[['CVR', 'FromDate', 'EmployeeCounts']]
            df_employees['FromDate'] = pd.to_datetime(df_employees['FromDate'])

            # Filter away data before 2013
            df_employees = df_employees.loc[df_employees['FromDate'] >= datetime(2013, 1, 1)]

            # read chunk of registration data 
            df_registrations = pd.read_csv(self.input_csv  / f"Registrations/chunk{i}.csv", index_col=0)

            # filter away CVR's that are not in the lookup table from the employee data and registration data
            df_employees = df_employees.loc[df_employees['CVR'].isin(df_cvr['CVR'].values)]
            df_registrations = df_registrations.loc[df_registrations['CVR'].isin(df_cvr['CVR'].values)]

            # merge
            df_employees = enrich_with_asof_values(df_employees, df_registrations)

            # Convert to Dask DataFrame and append to list
            ddf_list.append(
                        dd.from_pandas(df_employees, 
                                        npartitions=1)
                        )
            
        # Concatenate all DataFrames in the list
        ddf = dd.concat(ddf_list, axis=0).rename(columns = {
                    'CompanyType' : 'COMPANY_TYPE',
                    'Industry' : 'INDUSTRY',
                    'Status' : 'COMPANY_STATUS', 
                    'Municipality' : 'MUNICIPALITY', 
                    'FromDate' : 'FROM_DATE',
                    'EmployeeCounts' : 'EMPLOYEE_COUNT'
                })
        
  

        del ddf_list


        # Handle missing values and deal with datatypes
        ddf = ddf.fillna({
            'COMPANY_TYPE': '[UNK]',
            'INDUSTRY': '[UNK]',
            'COMPANY_STATUS': '[UNK]',
            'MUNICIPALITY': '[UNK]',
            'EMPLOYEE_COUNT': '[UNK]'
        })
        
        assert isinstance(ddf, dd.DataFrame)

        return ddf
    

# #use for debugging
# if __name__ == "__main__":
#     tokens = EmployeeTokens()
#     parsed_data = tokens.tokenized()
    