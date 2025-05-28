#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import logging
import time
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, List, Sequence

import hydra
import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.logging import RichHandler
from rich.prompt import Prompt


# ---------------------------------- Logger Setup ----------------------------------
def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes a Python command line logger."""
    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%m/%d %H:%M:%S]", handlers=[RichHandler()]
    )
    return logging.getLogger(name)


log = get_pylogger(__name__)


# ---------------------------------- Config Printing ----------------------------------
@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "datamodule",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints DictConfig content using Rich library.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Order of config components to print.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        save_to_file (bool, optional): Whether to export config to the hydra output folder.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)
    queue = [field for field in print_order if field in cfg] + [
        field for field in cfg if field not in print_order
    ]

    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)
        config_group = cfg[field]
        branch_content = (
            OmegaConf.to_yaml(config_group, resolve=resolve)
            if isinstance(config_group, DictConfig)
            else str(config_group)
        )
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user for tags if none are provided in config."""
    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        log.warning("No tags provided in config. Prompting user for input...")
        tags = [
            t.strip()
            for t in Prompt.ask("Enter a list of comma-separated tags", default="dev").split(",")
            if t
        ]

        with open_dict(cfg):
            cfg.tags = tags

        log.info(f"Tags: {cfg.tags}")

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.log"), "w") as file:
            rich.print(cfg.tags, file=file)


# ---------------------------------- Task Wrapper ----------------------------------
def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps a task function with additional utilities.

    Utilities:
    - Calls `extras(cfg)` before the task starts.
    - Calls `close_loggers()` after task completion.
    - Logs exceptions if they occur.
    - Logs execution time.
    - Logs output directory.

    Args:
        task_func (Callable): The function to wrap.

    Returns:
        Callable: The wrapped function.
    """

    def wrap(cfg: DictConfig):
        extras(cfg)

        try:
            start_time = time.time()
            metric_dict, object_dict = task_func(cfg=cfg)
        except Exception as ex:
            log.exception("")  # Save exception to `.log` file
            raise ex
        finally:
            path = Path(cfg.paths.output_dir, "exec_time.log")
            content = f"'{cfg.task_name}' execution time: {time.time() - start_time} (s)"
            save_file(path, content)
            close_loggers()

        log.info(f"Output dir: {cfg.paths.output_dir}")
        return metric_dict, object_dict

    return wrap


# ---------------------------------- Extra Utilities ----------------------------------
def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before task execution.

    - Ignores Python warnings.
    - Prompts user for tags if none are provided.
    - Prints the config tree.

    Args:
        cfg (DictConfig): Hydra configuration.
    """
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling Python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        enforce_tags(cfg, save_to_file=True)

    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        print_config_tree(cfg, resolve=True, save_to_file=True)


# ---------------------------------- Utility Functions ----------------------------------
@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Saves content to a file (rank zero only)."""
    with open(path, "w+") as file:
        file.write(content)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    if not callbacks_cfg:
        log.warning("Callbacks config is empty.")
        return []

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    return [
        hydra.utils.instantiate(cb_conf)
        for _, cb_conf in callbacks_cfg.items()
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf
    ]


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""
    if not logger_cfg:
        log.warning("Logger config is empty.")
        return []

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    return [
        hydra.utils.instantiate(lg_conf)
        for _, lg_conf in logger_cfg.items()
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf
    ]


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Logs hyperparameters for PyTorch Lightning.

    Additionally logs:
    - Number of model parameters.

    Args:
        object_dict (dict): Dictionary containing Hydra objects.
    """
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    model = object_dict["model"]
    cfg = object_dict["cfg"]

    hparams = {
        "model": cfg["model"],
        "model/params/total": sum(p.numel() for p in model.parameters()),
        "model/params/trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model/params/non_trainable": sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        ),
        "datamodule": cfg["datamodule"],
        "trainer": cfg["trainer"],
        "callbacks": cfg.get("callbacks"),
        "extras": cfg.get("extras"),
        "task_name": cfg.get("task_name"),
        "tags": cfg.get("tags"),
        "ckpt_path": cfg.get("ckpt_path"),
        "seed": cfg.get("seed"),
    }

    trainer.logger.log_hyperparams(hparams)


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Retrieves a logged metric value from LightningModule."""
    if not metric_name:
        log.info("Metric name is None! Skipping metric retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric '{metric_name}' not found! Ensure it is logged in LightningModule."
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value: {metric_name} = {metric_value}")
    return metric_value


def close_loggers() -> None:
    """Closes all loggers to prevent failures in multi-run scenarios."""
    log.info("Closing loggers...")
    if find_spec("wandb"):
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()
