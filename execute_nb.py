import os
from glob import *
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from sumatra.projects import load_project

nb_name = sys.argv[-1]
current_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.abspath(load_project().data_store.root)

src_nb_path = os.path.join(current_path, nb_name, nb_name+".ipynb")
output_nb_path = os.path.join(data_path, nb_name+".ipynb")

print("src_nb_path:", src_nb_path)
print("output_nb_path:", output_nb_path)

# execute nb
with open(src_nb_path) as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=-1)
ep.preprocess(nb, {"metadata": {"path": nb_name}})

with open(output_nb_path, 'wt') as f:
    nbformat.write(nb, f)

# convert to rst with corresponding figs
os.system("jupyter nbconvert --to rst {}".format(output_nb_path))

# convert pdf to eps
file_list = glob(os.path.join(data_path, nb_name+"_files", "*.pdf"))
for f in file_list:
    os.system('pdftops -eps {0}'.format(f))
