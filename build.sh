#!/bin/sh

if [ -z "$CONDA_PREFIX_1" ]; then
  source $CONDA_PREFIX/etc/profile.d/conda.sh
fi
conda activate pedroleitao.nl
quarto render --log build.log --log-level info
