# source me from the repository root: sets HF_TOKEN (only that line of .env), paths, python
# (override the root with SPECDEC_ROOT=/path/to/repo if sourcing from elsewhere)
M=${SPECDEC_ROOT:-$PWD}
export HF_TOKEN=$(grep '^HF_TOKEN=' $M/.env | cut -d= -f2- | tr -d '"'"'")
export UV_PROJECT_ENVIRONMENT=.venv-mac
D=$M/data/open-swe-traces
PY=($M/.venv-mac/bin/python -I)   # zsh array: use as "${PY[@]}" or $PY
X=$M/experiments/open-swe-traces-audit
REV=4fa4c2ae9afc6b397bc50391d43450c79993c138
