#!/usr/bin/env python3

import argparse
import os.path
import subprocess

import config

# from dsbox_dev_setup import path_setup
# path_setup()

import dsbox

parser = argparse.ArgumentParser(
    description='Generate primitive.json descriptions')
parser.add_argument(
    'dirname', action='store', help='Top-level directory to store the json descriptions')
arguments = parser.parse_args()

PREFIX = 'd3m.primitives.'
PRIMITIVES = [(p, config) for p in [
        'feature_construction.corex_continuous.CorexContinuous',
        'feature_construction.corex_text.CorexText',
        'regression.corex_supervised.EchoLinear',
        'feature_construction.corex_supervised.EchoIBReg',
        'feature_construction.corex_supervised.EchoIBClf'
]
]

for p, config in PRIMITIVES:
    print('Generating json for primitive ' + p)
    primitive_name = PREFIX + p
    outdir = os.path.join(arguments.dirname, 'v' + config.D3M_API_VERSION,
                          config.D3M_PERFORMER_TEAM, primitive_name,
                          config.VERSION)
    subprocess.run(['mkdir', '-p', outdir])

    json_filename = os.path.join(outdir, 'primitive.json')
    print('    at ' + json_filename)
    command = ['python', '-m', 'd3m.index',
               'describe', '-i', '4', primitive_name]
    with open(json_filename, 'w') as out:
        subprocess.run(command, stdout=out)
