from dask.distributed import Client
import hydra
from hydra.utils import instantiate

import os
from omegaconf import OmegaConf
import dask
import pandas as pd
from datetime import datetime


import logging
log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="prepare_data", version_base=None)
def main(cfg):
    
    target = instantiate(cfg.target, _convert_="all")
    
    if cfg.single_threaded:
        dask.config.set(scheduler="single-threaded") 
        target.prepare()
    else:
        client: Client = instantiate(cfg.client)
        log.info("Monitor progress at: %s", client.dashboard_link)
        with client:
            target.prepare()

    if cfg.interact:
        import code
        code.interact(local=locals())


if __name__ == "__main__":
    main()