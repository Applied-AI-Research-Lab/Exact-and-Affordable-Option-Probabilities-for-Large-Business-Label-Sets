#!/bin/tcsh -f
if (! $?PROJECT_ROOT) setenv PROJECT_ROOT `pwd`
cd $PROJECT_ROOT
setenv UV_TORCH_BACKEND cu129
mkdir -p logs

if (! -d .venv) uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt || exit 1

if (! -d .venv-train) uv venv --python 3.12 .venv-train
uv pip install --python .venv-train/bin/python -r requirements-train.txt || exit 1

if (! -d .venv-clef) uv venv --python 3.12 .venv-clef
uv pip install --python .venv-clef/bin/python -r requirements-clef.txt || exit 1

rehash
