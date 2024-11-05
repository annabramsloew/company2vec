from dataclasses import dataclass
from functools import reduce
from typing import Tuple

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

from ..decorators import save_parquet, save_pickle
from ..ops import concat_sorted
from ..serialize import DATA_ROOT
from ..source.financials import AnnualReportTokens
from .base import DataSplit, Population

nunique = dd.Aggregation(
    name="nunique",
    chunk=lambda s: s.apply(lambda x: list(set(x))),
    agg=lambda s0: s0.obj.groupby(level=list(range(s0.obj.index.nlevels))).sum(),
    finalize=lambda s1: s1.apply(lambda final: len(set(final))),
)

@dataclass
class FromAnnualReports(Population):
    """
    A cohord defined based on the labour dataset.

    :param labour_data: Instance of :class:`src.data.sources.LabourTokens` token source
        to base the population on.
    :param earliest_birthday: Earliest allowed birthday
    :param latest_birthday: Latest allowed birthday
    :param seed: Seed for splitting training, validation and test dataset
    :param train_val_test: Fraction of the data to be included in the three data splits.
        Must sum to 1.
    """

    token_data: AnnualReportTokens
    name: str = "annual_reports_test1"
    seed: int = 123
    train_val_test: Tuple[float, float, float] = (0.7, 0.15, 0.15)

    def __post_init__(self) -> None:
        assert sum(self.train_val_test) == 1.0


    @save_pickle(
        DATA_ROOT / "processed/populations/{self.name}/population",
        on_validation_error="error",
    )

    def population(self) -> pd.DataFrame:
        """Loads the combined annual report data.
        Currently no filters are applied. 
        """ #TODO: consider whether to use company info table, or maybe a filtered version of the registrations table?

        result = self.combined().compute()

        assert isinstance(result, pd.DataFrame)
        return result

    @save_parquet(
        DATA_ROOT / "interim/populations/{self.name}/combined",
        on_validation_error="recompute",
    )
    def combined(self) -> dd.DataFrame:
        """
        Pulls out the CVR (the index)
        """

        ls = self.token_data
        result = ls.indexed().drop(columns=['ASSETS', 'LIABILITIES_AND_EQUITY', 'EQUITY', 'PROFIT_LOSS', "MUNICIPALITY", 'INDUSTRY', 'COMPANY_TYPE', 'COMPANY_STATUS'])
        # Reset index and drop duplicates based on the CVR column
        result = result.reset_index().drop_duplicates(subset="CVR")

        # Sort the DataFrame by CVR
        result = result.sort_values(by="CVR")

        # Set CVR as the index
        result = result.set_index("CVR")

        # Repartition the DataFrame to ensure sorted partitions
        result = result.repartition(npartitions=result.npartitions)
        assert isinstance(result, dd.DataFrame)
        return result

    @save_pickle(DATA_ROOT / "processed/populations/{self.name}/data_split")
    def data_split(self) -> DataSplit:
        """Split data based on :attr:`seed` using :attr:`train_val_test` as ratios"""
        ids = self.population().index.to_numpy()
        np.random.default_rng(self.seed).shuffle(ids)
        split_idxs = np.round(np.cumsum(self.train_val_test) * len(ids))[:2].astype(int)
        train_ids, val_ids, test_ids = np.split(ids, split_idxs)
        return DataSplit(
            train=train_ids,
            val=val_ids,
            test=test_ids,
        )


# if __name__ == "__main__":
#     population = FromAnnualReports()
#     data_population = population.population()
#     data_split = population.data_split()
