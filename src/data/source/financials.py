# # Adapted from: https://github.com/SocialComplexityLab/life2vec/blob/v1.0.0/src/data_new/sources/education.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import dask.dataframe as dd
import pandas as pd
import gc

from ..decorators import save_parquet
from ..ops import sort_partitions
from ..logging_config import DATA_ROOT
from .base import FIELD_TYPE, TokenSource, Binned
from .source_helpers import dd_enrich_with_asof_values, convert_currency


@dataclass
class AnnualReportTokens(TokenSource):
    """This generates tokens based on information from the annual reports, registrations and currency datasets.
    Currently loads data from a CSV dump of the annual reports.

    :param input_csv: path to the Tables folder, from which data on registrations, currency rates and annual reports may be fetched.
    :param earliest_start: The earliest start date of an event
    """

    name: str = "annual_report"
    fields: List[FIELD_TYPE] = field(
        default_factory=lambda: [
            "COMPANY_TYPE", 
            "INDUSTRY", 
            "COMPANY_STATUS", 
            "MUNICIPALITY", 
            Binned("PROFIT_LOSS", prefix="PROFIT_LOSS", n_bins=100),
            Binned("EQUITY", prefix="EQUITY", n_bins=100),
            Binned("ASSETS", prefix="ASSETS", n_bins=100),
            Binned("LIABILITIES_AND_EQUITY", prefix="LIABILITIES_AND_EQUITY", n_bins=100)
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
                COMPANY_TYPE=lambda x: x.COMPANY_TYPE.map({"A/S": "CTYP_AS", 
                                                            "APS": "CTYP_APS", 
                                                            "IVS": "CTYP_IVS",
                                                            "[UNK]": "[UNK]"}, meta=('COMPANY_TYPE', 'object')),

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
                                                                "[UNK]" : "[UNK]"
                                                                }, meta=('COMPANY_STATUS', 'object')), 
                MUNICIPALITY=lambda x: x.MUNICIPALITY.apply(lambda mun: "MUN_" + mun if mun != "[UNK]" else mun, meta=('MUNICIPALITY', 'object'))
            ).pipe(sort_partitions, columns=["FROM_DATE"])[["FROM_DATE", *self.field_labels()]]
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

        columns_registrations = ["CVR", "FromDate", "ChangeType", "NewValue"]

        columns_annualreport = [
            "CVR",
            "PublicationDate",
            "Currency",
            "ProfitLoss",
            "Equity", 
            "Assets", 
            "LiabilitiesAndEquity"
        ]

        columns_currency = ["year", "month", "from_currency", "rate"]

        output_columns = [
            "FROM_DATE",
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
        path_cvr = self.input_csv / "CVRFiltered"
        
        # Load files
        financials_csv = [file for file in path_financials.iterdir() if file.is_file() and file.suffix == '.csv']
        registrations_csv = [file for file in path_registrations.iterdir() if file.is_file() and file.suffix == '.csv']
        currency_csv = [file for file in path_currency.iterdir() if file.is_file() and file.suffix == '.csv']
        cvr_csv = [file for file in path_cvr.iterdir() if file.is_file() and file.suffix == '.csv']

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
        ).compute()





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
        ).compute()
        
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

        df_cvr = dd.read_csv(
            cvr_csv,
            usecols=['CVR'],
            dtype={
                "CVR": int
            }
        ).repartition(npartitions=2)
        

        # filter away CVR's that are not in the lookup table from the annual report data and registration data
        cvr_list = df_cvr['CVR'].compute()#.tolist()#[:1000]
        ddf_annualreport = ddf_annualreport.loc[ddf_annualreport['CVR'].isin(cvr_list)]
        ddf_registrations = ddf_registrations.loc[ddf_registrations['CVR'].isin(cvr_list)]


        print("START ENRICH")
        # enrich the annual report with asof values from the registrations
        ddf = dd_enrich_with_asof_values(
            ddf_annualreport, 
            ddf_registrations, 
            values=['Industry', 'CompanyType', 'Municipality', 'Status'], 
            date_col_df='PublicationDate', 
            date_col_registrations='FromDate'
            )
        
        ddf = dd.from_pandas(ddf)
        print("SAVED")
        
        # convert currency
        ddf = convert_currency(ddf, ddf_currency, 
                        amount_cols=['ProfitLoss', 'Equity', 'Assets', 'LiabilitiesAndEquity'], 
                        currency_col='Currency', 
                        date_col='PublicationDate')

        
        # Drop 'Currency' column, rename columns, and drop values
        column_map = {
            "PublicationDate":"FROM_DATE",
            "ProfitLoss":"PROFIT_LOSS",
            "Assets":"ASSETS",
            "Equity":"EQUITY",
            "LiabilitiesAndEquity":"LIABILITIES_AND_EQUITY",
            "Industry":"INDUSTRY",
            "CompanyType":"COMPANY_TYPE",
            "Municipality":"MUNICIPALITY",
            "Status":"COMPANY_STATUS"
        }

        
        ddf = (
            ddf.drop(columns=['Currency'])
            .rename(columns=column_map)
            .loc[lambda x: x.FROM_DATE >= self.earliest_start]
        )

        ddf = ddf.fillna({
            'COMPANY_TYPE': '[UNK]',
            'INDUSTRY': '[UNK]',
            'COMPANY_STATUS': '[UNK]',
            'MUNICIPALITY': '[UNK]',
        })


        if self.downsample:
            ddf = self.downsample_persons(ddf)

        assert isinstance(ddf, dd.DataFrame)

        return ddf
    

# use for debugging
# if __name__ == "__main__":
#     tokens = AnnualReportTokens()
#     parsed_data = tokens.tokenized()
#     print("FINISHED")
    