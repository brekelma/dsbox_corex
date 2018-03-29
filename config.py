import os
from d3m_metadata import utils

D3M_API_VERSION = '2018.1.26'
VERSION = "1.0.0"
TAG_NAME = "{git_commit}".format(git_commit=utils.current_git_commit(os.path.dirname(__file__)), )

REPOSITORY = "https://github.com/brekelma/dsbox_corex"
PACKAGE_NAME = "dsbox_corex"

D3M_PERFORMER_TEAM = 'ISI'
MAIL_TO = "mailto:brekelma@usc.edu"

if TAG_NAME:
    PACKAGE_URI = "git+" + REPOSITORY + "@" + TAG_NAME
else:
    PACKAGE_URI = "git+" + REPOSITORY 

PACKAGE_URI = PACKAGE_URI + "#egg=" + PACKAGE_NAME


INSTALLATION_TYPE = 'GIT'
if INSTALLATION_TYPE == 'PYPI':
    INSTALLATION = {
        "type" : "PIP",
        "package": PACKAGE_NAME,
        "version": VERSION
    }
else:
    # INSTALLATION_TYPE == 'GIT'
    INSTALLATION = {
        "type" : "PIP",
        "package_uri": PACKAGE_URI,
    }