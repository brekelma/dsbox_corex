# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

try:
    long_description = open("README.rst").read()
except IOError:
    long_description = ""

setup(
    name="dsbox_corex",
    version="1.0.0",
    description="Return components/latent factors that explain the most multivariate mutual information in the data under Linear Gaussian model. For comparison, PCA returns components explaining the most variance in the data.",
    license="AGPL-3.0",
    author="Rob Brekelmans/Greg Ver Steeg",
    author_email="brekelma@usc.edu",
    keywords='d3m_primitive',
    #packages = ['corexcontinuous', 'corextext', ]
    packages=find_packages(),
    url='https://github.com/brekelma/dsbox_corex',
    download_url='https://github.com/brekelma/dsbox_corex',
    install_requires=[],
    long_description=long_description,
    include_package_data = True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
    ], 
    entry_points = {
    'd3m.primitives': [
        'feature_construction.corex_continuous.DSBOX = corex_continuous:CorexContinuous',
        'feature_construction.corex_text.DSBOX = corex_text:CorexText',
        #'regression.corex_supervised.DSBOX = echo_regressor:EchoLinearRegression',
        'regression.corex_supervised.EchoLinear = echo_regressor:EchoLinearRegression',
        'feature_construction.corex_supervised.EchoIB = echo_ib:EchoIB'
        #'feature_construction.corex_supervised.EchoIBReg = echo_sae:EchoRegression',        
        #'feature_construction.corex_supervised.EchoIBClf = echo_sae:EchoClassification'
    ]
    }
)
