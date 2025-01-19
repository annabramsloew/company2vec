from abc import ABC, abstractmethod
from ast import Str
from posixpath import split
from sys import meta_path
import numpy as np
import time
import os
import pickle
from typing import Any, List, Tuple, Callable, Dict, NamedTuple
from functools import reduce
import warnings
from torch.nn import Module
import torch
from captum.attr import LayerActivation
import re
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
import sys
from pathlib import Path
import logging

def last_ckpt(dir_):
    ckpt_path = Path(HOME_PATH, dir_, "last.ckpt")
    if ckpt_path.exists():
        log.info("Checkpoint exists:\n\t%s" %str(ckpt_path))
        return str(ckpt_path)
    else:
        log.info("Checkpoint DOES NOT exists:\n\t%s" %str(ckpt_path))
        return None

def home_path():
    return HOME_PATH

log = logging.getLogger(__name__)
HOME_PATH = str(Path.home())
ATTR_PATH = r"/zhome/49/1/147319/odrev/projekter/PY000017_D/activations/bankruptcy/"

class BaseActivations(ABC):
    """Base class for computing the company summary activations"""
    @abstractmethod
    def get_concept_activations(self, calculate):
        NotImplemented

    def get_dataloader(self, dataset, indices):
        return self.data.get_fixed_dataloader(dataset, indices)

    def calculate_activations(self, data_loader):
        """Returns the activations for a each item after a given layer
        Args:
            data_iterator: Iterator that iterates over samples
        """
        layer_modules = [get_module_from_name(self.model, l) for l in self.layer] 
        print(layer_modules)
        layer_act = LayerActivation(self.model.forward_with_embeddings, layer_modules)
        print(layer_act)
        activations  = list() 
        iteration = 0
        for batch in data_loader:
            log.info(iteration)
            iteration += 1
            embeddings, _ = self.model.transformer.get_sequence_embedding(batch["input_ids"].long().to(self.device))
            embeddings = embeddings.detach()
            act = layer_act.attribute(embeddings, additional_forward_args={"padding_mask": batch["padding_mask"].long().to(self.device)}, attribute_to_layer_input=True)
            activations.append(act[0].detach().cpu().numpy())

        return np.vstack(activations)



class Activations(BaseActivations):
    """Produces Activations for random samples in TEST SET"""
    def __init__(self, 
                version: str,
                model: Tuple[Module, None],
                data,
                layers: List[str],
                sample_idx = None,
                custom_dataloader = None,
                random_seed: int = 2021):

        self.random_seed = random_seed
        self.version = version

        self.layer = layers
        self.model = model
        self.data = data
        
        assert len(layers) == 1

        if sample_idx is None:
            sample_idx = self.index_sample_data(self.data, sample_size=10000, random_seed=random_seed)

        if custom_dataloader is None:   
            self.loader_samples = data.get_fixed_dataloader(self.data.test, sample_idx)
        else: 
            self.loader_samples = custom_dataloader

        if self.model is None:
            warnings.warn("Model is not provided, use load_model method to assign a nn.Module")  
        else:
            self.device = model.device
            self.model.eval()

    @staticmethod
    def summarize_attr(attr, mask):
        attr = attr.sum(dim=-1)
        attr = attr[mask]
        return attr/torch.norm(attr)

    def get_concept_activations(self, calculate: bool = False):
        output_folder = ATTR_PATH + "sample_act/%s_%s" %(self.version, self.layer[0])
        output_path = output_folder + "/act.pkl"
        if calculate:
            print("==== ACTIVATIONS START =====")
            activations = self.calculate_activations(self.loader_samples)
            try:
                os.makedirs(output_folder, exist_ok=False)
            except:
                pass
            with open(output_path, "wb") as f:
                pickle.dump(activations, f)
            print("==== ACTIVATIONS CALCULATED =====")
            print("\t Results saved to", output_path)
        with open(output_path, "rb") as f:
            activations = pickle.load(f)
        return activations

    def get_concept_metadata(self, calculate):
        output_folder = ATTR_PATH + "sample_meta/%s_%s" %(self.version, self.layer[0])
        output_path = output_folder + "/meta.pkl" 
        if calculate:
            sequence = []
            sequence_ids = []
            targets = []
            preds = []
            print("==== METADATA START =====")
            for batch in self.loader_samples:

                for k in batch.keys():
                    torch.cuda.empty_cache()
                    batch[k] = batch[k].to(self.model.device)
                preds.append(self.model(batch).detach().cpu().numpy())
                sequence_ids.extend(batch["sequence_id"].tolist())
                sequence.append(batch["input_ids"][:,0].detach().cpu().numpy())
                targets.extend(batch["target"].tolist())
            try:
                os.makedirs(output_folder, exist_ok=False)
            except:
                pass
            with open(output_path, "wb") as f:
                pickle.dump({"metadata": np.concatenate(sequence), "sequence_ids": sequence_ids, "targets": targets, "predictions": np.concatenate(preds)}, f)
            print("==== METADATA COLLECTED =====")
            print("\t Results saved to", output_path)
        with open(output_path, "rb") as f:
            metadata = pickle.load(f)
        return metadata

    def index_sample_data(self, dataset, sample_size, random_seed=2021):
        indices = dataset.get_ordered_indexes(split = "test")
        print(indices)
        np.random.seed(random_seed)
        dataset_size = len(indices)
        return np.random.choice(np.arange(dataset_size), size=sample_size, replace=False)

def get_module_from_name(model, layer_name: str) -> Any:
    """
    Returns the module (layer) object, given its (string) name
    in the model.
    Args:
        name (str): Module or nested modules name string in self.model

    Returns:
        The module (layer) in self.model.
    """
    return reduce(getattr, layer_name.split("."), model)

try:
    OmegaConf.register_new_resolver("last_ckpt", last_ckpt)
except Exception as e:
    print(e)

@hydra.main(config_path="../conf", config_name="config")
def main(cfg):

    ##GLOBAL SEED
    seed_everything(cfg.seed)
    
    data = instantiate(cfg.datamodule, _convert_="all")
    data.setup()

    ##MODEL
    model = instantiate(cfg.model, _convert_="all")
    

    layers = ['decoder.identity']  
    activations = Activations(
        version='1.0',
        model=model,
        data=data,
        layers=layers
    )
    print("calculating concept activations")
    activations.get_concept_activations(calculate=True)
    print("calculating concept metadata")
    activations.get_concept_metadata(calculate=True)


if __name__ == "__main__":
    main()

