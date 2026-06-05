from __future__ import annotations

import logging


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def info(message: str) -> None:
    logging.getLogger("strata_fit_v6_meta_algo_py").info(message)


def warn(message: str) -> None:
    logging.getLogger("strata_fit_v6_meta_algo_py").warning(message)


def error(message: str) -> None:
    logging.getLogger("strata_fit_v6_meta_algo_py").error(message)
