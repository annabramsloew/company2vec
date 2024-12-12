from dataclasses import dataclass
from doctest import ELLIPSIS_MARKER
from itertools import chain
from typing import List, TypeVar, cast
from datetime import date

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.data.types import Background, JSONSerializable, CompanyDocument, EncodedDocument
from src.tasks.base import Task

from functools import reduce
import logging

T = TypeVar("T")
log = logging.getLogger(__name__)



@dataclass
class TabularCLS(Task):

    def register(self, datamodule) -> None:
        self.datamodule = datamodule
  
        vocab = self.datamodule.vocabulary.vocab()
        vocab = vocab[~vocab.CATEGORY.isin(["GENERAL", "MONTH", "BACKGROUND", "YEAR"])]
        vocab = vocab["TOKEN"].tolist()
        log.info("Input size: %s" %(len(vocab) + 5))
        self.vectorizer = CountVectorizer(vocabulary=vocab,  token_pattern=r"\S+", lowercase=False)
        self.period_start = self.datamodule.corpus.population._period_start
        self.period_end = self.datamodule.corpus.population._period_end

    # CLS Specific params
    def get_document(self, company_sentences: pd.DataFrame) -> CompanyDocument:
        document = super().get_document(company_sentences)
        target = int(company_sentences.TARGET.iloc[0])
        document.task_info = cast(JSONSerializable, target)
        return document

    def encode_document(self, document: CompanyDocument) -> "TCLSEncodedDocument":

        origin_dk  = 1. if document.background.origin == "DK" else 0.
        origin_nd  =  1. - origin_dk
        background = np.array([origin_dk, origin_nd], dtype=np.float32) #deleted , sex_m, sex_f, age from brackets
        if len(document.sentences) == 0:
            sequence = "[UNK]"
        else: 
            sequence = " ".join(reduce(lambda xs, ys: xs + ys, document.sentences))
        data = self.vectorizer.transform([sequence]).toarray().flatten().astype(np.float32)
        
        data /= data.sum()

        data = np.concatenate([background, data])


        target = np.array(document.task_info).astype(int)

        sequence_id = np.array(document.cvr)


        return TCLSEncodedDocument(
            sequence_id=sequence_id,
            input_ids=data,
            target=target,
            background=background,
        )

@dataclass
class TCLSEncodedDocument(EncodedDocument[TabularCLS]):
    sequence_id: np.ndarray
    input_ids: np.ndarray
    target: np.ndarray
    background: np.ndarray