import os
import re

def get_version():
    VERSIONFILE = os.path.join('paradime', '_version.py')
    version_lines = open(VERSIONFILE, 'rt').readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in version_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError(f"Unable to find version string in {VERSIONFILE}.")

# ...

# setup(
#         version = get_version()
# )

# TODO: read up on pyproject.toml (maybe remove setup.py)
# remaining question: how to populate version automatically?