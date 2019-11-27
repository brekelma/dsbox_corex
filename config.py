import os
import d3m
from d3m import utils

D3M_API_VERSION = '2019.11.10' #d3m.__version__
VERSION = "1.0.0"
TAG_NAME = "{git_commit}".format(git_commit=utils.current_git_commit(os.path.dirname(__file__)), )

REPOSITORY = "https://github.com/brekelma/dsbox_corex"
PACAKGE_NAME = "dsbox-corex"

D3M_PERFORMER_TEAM = 'ISI'

if TAG_NAME:
    PACKAGE_URI = "git+" + REPOSITORY + "@" + TAG_NAME
else:
    PACKAGE_URI = "git+" + REPOSITORY 

PACKAGE_URI = PACKAGE_URI + "#egg=" + PACAKGE_NAME


INSTALLATION_TYPE = 'GIT'
if INSTALLATION_TYPE == 'PYPI':
    INSTALLATION = {
        "type" : "PIP",
        "package": PACAKGE_NAME,
        "version": VERSION
    }
else:
    INSTALLATION = {
        "type" : "PIP",
        "package_uri": PACKAGE_URI,
    }
