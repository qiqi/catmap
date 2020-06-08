import os
import sys
import json
import subprocess


myPath = os.path.dirname(os.path.abspath(__file__))
cmd = 'python3 {}/{}'.format(myPath, sys.argv[1])
config = json.load(open(os.path.join(myPath, 'bakers.json')))
nodes = config['nodes']

procs = [subprocess.Popen(['ssh', n, cmd]) for n in nodes]
for p in procs:
    p.wait()
