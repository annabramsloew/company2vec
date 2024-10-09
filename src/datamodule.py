# import enum
# import logging
# import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import dataclass_transform
from .logging_config import log
import dask
import dask.dataframe as dd
import pandas as pd
import numpy as np
# import pytorch_lightning as pl
# import torch
# from pandas.tseries.offsets import MonthEnd
# from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# from ..tasks.base import Task, collate_encoded_documents
# from .sampler import FixedSampler
# from .dataset import DocumentDataset, ShardedDocumentDataset
from .decorators import save_parquet, save_pickle
from .ops import concat_columns_dask, concat_sorted
# from .populations.base import Population
from .serialize import DATA_ROOT, ValidationError, _jsonify
from .source.base import Field, TokenSource
from .source.employees import EmployeeTokens
from .source.punits import ProductionUnitTokens
from .source.capital import CapitalTokens
# from .vocabulary import Vocabulary

N_PARTITIONS = 43


@dataclass
class Corpus:
    """
    Provides a corpus for the specified population with tokens for the specified
    sources. Splits the data into training, validation and testing partition according
    to the population splits.

    .. todo::
        consider renaming the reference_date and threshold parameters to something more
        meaningful

    :param sources: List of token sources from which to generate sentences
    :param population: Cohort to generate sentences for.

    :param reference_date:
    :param threshold:

    """

    name: str

    sources: List[TokenSource]
    #TODO: population: Population

    reference_date: str = "2008-01-01" 
    threshold: str = "2016-01-01"
    population: Optional[Any] = None # TODO: remove for actual population

    def __post_init__(self) -> None:

        self._reference_date = pd.to_datetime(self.reference_date)
        self._threshold = pd.to_datetime(self.threshold)

    @save_parquet(DATA_ROOT / "processed/corpus/{self.name}/sentences/{split}")
    def combined_sentences(self, split: str) -> dd.DataFrame:
        """Combines the sentences from each source. Filters the data to only consist
        of sentences for the given :obj:`split`.

        :param split: Data split to return sentences for.

        :return:

            A :class:`dask.dataframe.DataFrame` object with the following columns

            * PERSON_ID (Index column) - The person ids.

            * START_DATE - Date of sentence as number of days since
              :attr:`self.reference_date`

            * SENTENCE - The sentence.

            * AGE - Is calculated bases on the birthday of each person. If the sentences
              already have an AGE columns, this is used instead.

            * GENDER - The gender as specified in the population data

            * AFTER_THRESHOLD - a boolean column, indicating whether an event is efter
              :attr:`self.threshold`.

            * Any additional columns from the population data is carried over as well.

        """

        #population: pd.DataFrame = self.population.population()
        #data_split = getattr(self.population.data_split(), split)



        sentences_parts = [self.sentences(s) for s in self.sources]
        combined_sentences = concat_sorted([sp for sp in sentences_parts], columns=["FROM_DATE"])

        # combined_sentences = concat_sorted(
        #     [sp.loc[lambda x: x.index.isin(data_split)] for sp in sentences_parts],
        #     columns=["START_DATE"],
        # ).join(population)

        # # Fix age from sources without age using birthday
        # isna = combined_sentences.AGE.isna()
        # combined_sentences["AGE"] = combined_sentences["AGE"].where(
        #     ~isna,
        #     compute_age(combined_sentences.START_DATE, combined_sentences.BIRTHDAY),
        # )

        # combined_sentences["AFTER_THRESHOLD"] = (
        #     combined_sentences.START_DATE >= self._threshold
        # )

        # # Date as days from reference date <- maybe move into task

        # combined_sentences["START_DATE"] = (
        #     combined_sentences.START_DATE - self._reference_date
        # ).dt.days.astype(int)

        ### DASK SPECIFIC
        combined_sentences = combined_sentences.reset_index().set_index("CVR", sorted=True)

        assert isinstance(combined_sentences, dd.DataFrame)

        return combined_sentences



    @save_parquet(
        DATA_ROOT / "interim/corpus/{self.name}/sentences_{source.name}",
        on_validation_error="recompute",
    )



    def sentences(self, source: TokenSource) -> dd.DataFrame:
        """Returns the sentences from :obj:`source`, ie all the fields in
        :attr:`source.fields` in the transformed tokenized data concatenated as strings.
        """
        tokenized = self.tokenized_and_transformed(source)
        field_labels = source.field_labels()

        import pandas.api.types as ptypes

        for field in field_labels:
            is_string = ptypes.is_string_dtype(tokenized[field].dtype)
            is_known_cat = (
                ptypes.is_categorical_dtype(tokenized[field].dtype)
                and tokenized[field].cat.known
            )
            assert is_string or is_known_cat

        cols = ["FROM_DATE", "SENTENCE"]
        
        # Ensure the DataFrame is properly partitioned
        tokenized = tokenized.repartition(npartitions=tokenized.npartitions)



        # It is a bit akwkard that we join, then split right after.
        # However it is easier to deal with strings, I think
        sentences = tokenized.astype({x: "string" for x in field_labels}).assign(
            SENTENCE=concat_columns_dask(tokenized, columns=list(field_labels))
        )[cols]

        assert isinstance(sentences, dd.DataFrame)

        return sentences

    @save_parquet(
        DATA_ROOT
        / "interim/corpus/{self.name}/tokenized_and_transformed/{source.name}",
        on_validation_error="recompute",
    )


    def tokenized_and_transformed(self, source: TokenSource) -> dd.DataFrame:
        """Returns the tokenized data for :obj:`source`, with any
        :class:`src.sources.base.Field` tranformations applied.
        """

        fields_to_transform = self.fitted_fields(source)
        tokenized = source.tokenized()
        tokenized = tokenized.repartition(npartitions=N_PARTITIONS)
        tokenized.to_parquet("test.parquet")


        for field in fields_to_transform:
            tokenized[field.field_label] = field.transform(tokenized[field.field_label])


        assert isinstance(tokenized, dd.DataFrame)
        return tokenized

    @save_pickle(
        DATA_ROOT / "interim/corpus/{self.name}/fitted_fields/{source.name}",
        on_validation_error="recompute",
    )


    def fitted_fields(self, source: TokenSource) -> List[Field]:
        """Fits any :class:`src.sources.base.Field` using the :meth:`fit` method on the
        training data, and saves their state using pickle.
        """
        # use only population ids id population has been initialized


        tokenized = source.tokenized()
        fields = source.fields
        fields_to_fit = [field for field in fields if isinstance(field, Field)]

        if self.population is not None:
            ids = self.population.data_split().train
            for field in fields_to_fit:
                field.fit(tokenized.loc[lambda x: x.index.isin(ids)])
        else: 
            for field in fields_to_fit:
                field.fit(tokenized)
                
        return fields_to_fit

    def prepare(self) -> None:
        """Prepares each dataset split"""

        self.combined_sentences("train")
        self.combined_sentences("val")
        self.combined_sentences("test")



if __name__ == "__main__":
    tokensources: List[TokenSource] = [CapitalTokens()]
    corpus = Corpus(name="test_capital", sources=tokensources)

    sentences = corpus.combined_sentences("train")
    
