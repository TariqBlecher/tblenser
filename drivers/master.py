import os

if os.path.exists('../lens_dir'):
    os.system('rm -r ../lens_dir')
os.system('mkdir ../lens_dir')
os.system('cp ./* ../lens_dir')
os.chdir('../lens_dir')
os.system('python cluster_statistics.py')