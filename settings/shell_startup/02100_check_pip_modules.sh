"$PROJECTR_FOLDER/settings/commands/.check_pip_modules"

# special local packages
python -m pip --disable-pip-version-check install -e ./source/modules/ml-agents-envs
python -m pip --disable-pip-version-check install -e ./source/modules/ml-agents
python -m pip --disable-pip-version-check install -e ./source/modules/gym-unity