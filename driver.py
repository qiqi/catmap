import os
import json
import subprocess

myPath = os.path.dirname(os.path.abspath(__file__))
cmd = 'python3 {}/cat.py'.format(myPath)
config = json.load(open(os.path.join(myPath, 'config.json')))
nodes = config['nodes']

procs = [subprocess.Popen(['ssh', n, cmd]) for n in nodes]
for p in procs:
    p.wait()
