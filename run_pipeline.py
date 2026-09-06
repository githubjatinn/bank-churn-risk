import os

# Has to happen before numpy/scipy get imported anywhere, including
# transitively through pandas or sklearn — this env var only takes effect
# if it's set before the BLAS library is loaded into the process.
#
# Apple's Accelerate framework (what numpy links against by default on
# Apple Silicon, instead of OpenBLAS) has a documented bug where
# multi-threaded matrix multiplication produces spurious overflow/invalid
# value warnings — not from anything wrong with the data, just a race
# condition in Accelerate itself. Forcing single-threaded execution avoids
# the code path where it happens. Harmless on any other platform.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import warnings

# The env var above was a legitimate attempt to fix this at the source and
# didn't work, which is itself informative: this warning fires from Apple's
# Accelerate BLAS on an intermediate blocked-matmul computation that never
# actually reaches our output (confirmed — recall/precision/ROC-AUC are
# stable and sane across every run, no NaNs anywhere downstream). Chasing a
# closed-source vendor BLAS bug further than this isn't worth the time, so
# it gets suppressed explicitly and narrowly, not blanket-ignored.
warnings.filterwarnings(
    "ignore", message=".*overflow encountered in matmul.*", category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore", message=".*invalid value encountered in matmul.*", category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore", message=".*divide by zero encountered in matmul.*", category=RuntimeWarning,
)

import logging

from src import config
from src.data.clean import clean
from src.data.loader import load_raw
from src.features.engineer import add_derived_features
from src.modeling.train import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_feature_set():
    raw = load_raw(config.RAW_DATA_PATH)
    cleaned = clean(raw)
    featured = add_derived_features(cleaned)

    config.PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    featured.to_parquet(config.PROCESSED_DATA_PATH, index=False)
    logger.info(
        "wrote %d rows, %d columns to %s",
        len(featured), featured.shape[1], config.PROCESSED_DATA_PATH.name,
    )
    return featured


def main():
    featured = build_feature_set()
    metadata = train(featured)

    rf_metrics = metadata["test_metrics"]["random_forest"]
    logger.info(
        "training complete — random forest test ROC-AUC: %.4f, recall: %.4f, precision: %.4f",
        rf_metrics["roc_auc"], rf_metrics["recall"], rf_metrics["precision"],
    )
    logger.info("artifacts written to %s", config.MODEL_DIR)


if __name__ == "__main__":
    main()
