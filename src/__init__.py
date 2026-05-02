# Marks `src/` as a Python package so that the per-architecture training
# entry points (train_resnet.py, train_efficientnet.py, train_mobilenet.py)
# can be invoked as modules from the repo root, e.g.:
#
#     python -m src.train_resnet --manifest configs/data_split.yaml
#
# and so they can do clean relative imports from sibling modules
# (data_prep, models, train, evaluate, gradcam, tune_optuna).
