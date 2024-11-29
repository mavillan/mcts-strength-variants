#!/bin/bash
set +e  # This tells bash to continue even if commands fails
################################################################################
python run.py train/nn-gandalf.ipynb --fs-type full
python run.py train/nn-gandalf.ipynb --fs-type fsv2
python run.py train/nn-gandalf.ipynb --fs-type fsv3
python run.py train/nn-gandalf.ipynb --fs-type fsv4
python run.py train/nn-gandalf.ipynb --fs-type fsv23
python run.py train/nn-gandalf.ipynb --fs-type fsv24
python run.py train/nn-gandalf.ipynb --fs-type fsv34
python run.py train/nn-gandalf.ipynb --fs-type uni80
python run.py train/nn-gandalf.ipynb --fs-type uni85
python run.py train/nn-gandalf.ipynb --fs-type uni90
python run.py train/nn-gandalf.ipynb --fs-type uni95
################################################################################
python run.py train/nn-1dcnn.ipynb --fs-type full
python run.py train/nn-1dcnn.ipynb --fs-type fsv2
python run.py train/nn-1dcnn.ipynb --fs-type fsv3
python run.py train/nn-1dcnn.ipynb --fs-type fsv4
python run.py train/nn-1dcnn.ipynb --fs-type fsv23
python run.py train/nn-1dcnn.ipynb --fs-type fsv24
python run.py train/nn-1dcnn.ipynb --fs-type fsv34
python run.py train/nn-1dcnn.ipynb --fs-type uni80
python run.py train/nn-1dcnn.ipynb --fs-type uni85
python run.py train/nn-1dcnn.ipynb --fs-type uni90
python run.py train/nn-1dcnn.ipynb --fs-type uni95
################################################################################
python run.py train/nn-mlp.ipynb --fs-type full
python run.py train/nn-mlp.ipynb --fs-type fsv2
python run.py train/nn-mlp.ipynb --fs-type fsv3
python run.py train/nn-mlp.ipynb --fs-type fsv4
python run.py train/nn-mlp.ipynb --fs-type fsv23
python run.py train/nn-mlp.ipynb --fs-type fsv24
python run.py train/nn-mlp.ipynb --fs-type fsv34
python run.py train/nn-mlp.ipynb --fs-type uni80
python run.py train/nn-mlp.ipynb --fs-type uni85
python run.py train/nn-mlp.ipynb --fs-type uni90
python run.py train/nn-mlp.ipynb --fs-type uni95
################################################################################
python run.py train/catboost.ipynb --fs-type full
python run.py train/catboost.ipynb --fs-type fsv2
python run.py train/catboost.ipynb --fs-type fsv3
python run.py train/catboost.ipynb --fs-type fsv4
python run.py train/catboost.ipynb --fs-type fsv23
python run.py train/catboost.ipynb --fs-type fsv24
python run.py train/catboost.ipynb --fs-type fsv34
python run.py train/catboost.ipynb --fs-type uni80
python run.py train/catboost.ipynb --fs-type uni85
python run.py train/catboost.ipynb --fs-type uni90
python run.py train/catboost.ipynb --fs-type uni95
################################################################################
python run.py train/catboost_text.ipynb --fs-type full
python run.py train/catboost_text.ipynb --fs-type uni90
################################################################################