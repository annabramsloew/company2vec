# # Adapted from: https://github.com/SocialComplexityLab/life2vec/blob/v1.0.0/src/data_new/sources/education.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

from datetime import datetime
import dask.dataframe as dd
import pandas as pd
import os

#from ..decorators import save_parquet
from ..ops import sort_partitions
#from ..serialize import DATA_ROOT
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
    
    input_csv: Path = Path(r"/Users/nikolaibeckjensen/Dropbox/Virk2Vec/Tables")
    earliest_start: str = "01/01/2008"

    
    def _post_init__(self) -> None:
        self._earliest_start = pd.to_datetime(self.earliest_start)


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
                COMPANY_STATUS=lambda x: "CSTAT_" + x.COMPANY_STATUS.map({
                                                                            "NORMAL": "ACTIVE",
                                                                            "AKTIV": "ACTIVE",
                                                                            "UNDER REASSUMERING" : "OTHER",
                                                                            "UNDER FRIVILLIG LIKVIDATION" : "OTHER",
                                                                            "UNDER REKONSTRUKTION" : "DISTRESS",
                                                                            "UNDER TVANGSOPLØSNING" : "DISTRESS",
                                                                            "UNDER KONKURS" : "DISTRESS",
                                                                            "OPLØST EFTER GRÆNSEOVERSKRIDENDE FUSION" : "DISSOLVED",
                                                                            "OPLØST EFTER GRÆNSEOVERSKRIDENDE HJEMSTEDSFLYTNING" : "DISSOLVED",
                                                                            "OPLØST EFTER SPALTNING" : "DISSOLVED",
                                                                            "OPLØST EFTER ERKLÆRING" : "DISSOLVED",
                                                                            "OPLØST EFTER FUSION" : "DISSOLVED",
                                                                            "OPLØST EFTER FRIVILLIG LIKVIDATION" : "DISSOLVED",
                                                                            "SLETTET" : "DISSOLVED",
                                                                            "OPLØST EFTER KONKURS" : "BANKRUPT",
                                                                            "TVANGSOPLØST" : "BANKRUPT",
                                                                        }), 
                MUNICIPALITY=lambda x: "WMUN_" + x.MUNICIPALITY, 
            )
            .pipe(sort_partitions, columns=["FROM_DATE"])[["FROM_DATE", *self.field_labels()]]
        )

        assert isinstance(result, dd.DataFrame)
        return result



    def indexed(self) -> dd.DataFrame:
        """Loads the parsed data, sets the index, then saves the indexed data"""
        result = self.parsed().set_index("CVR")
        assert isinstance(result, dd.DataFrame)
        return result



    def parsed(self) -> dd.DataFrame:
        """Parses the CSV file, applies some basic filtering, then saves the result
        as compressed parquet file, as this is easier to parse than the CSV for the
        next steps"""


        ddf_list = []
        for i in range(len(os.listdir(self.input_csv / "EmployeeCounts"))):
            print(i)

            # read chunk of employee data
            df_employees = pd.read_csv(self.input_csv / f"EmployeeCounts/chunk{i}.csv", index_col=0)[['CVR', 'FromDate', 'EmployeeCounts']]
            df_employees['FromDate'] = pd.to_datetime(df_employees['FromDate'])

            # Filter away data before 2013
            df_employees = df_employees.loc[df_employees['FromDate'] >= datetime(2013, 1, 1)]



            # read chunk of registration data 
            df_registrations = pd.read_csv(self.input_csv  / f"Registrations/chunk{i}.csv", index_col=0)

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


        # TODO: Drop missing values and deal with datatypes
        
        assert isinstance(ddf, dd.DataFrame)

        return ddf
    

# use for debugging
# if __name__ == "__main__":
#     tokens = EmployeeTokens()
#     parsed_data = tokens.tokenized().compute()
    