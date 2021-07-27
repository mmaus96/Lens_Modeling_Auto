import sys
if sys.platform[:5] == 'linux':
    import matplotlib
    matplotlib.use('Agg')
import os
from os import walk
from os import listdir
from os.path import isfile, join, isdir, exists
import re
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from lenstronomy.Plots import chain_plot
# from lenstronomy.Plots.chain_plot import plot_mcmc_behaviour
from copy import deepcopy
from lenstronomy.Util.param_util import ellipticity2phi_q
from Lens_Modeling_Auto.plot_functions import make_chainPlots
from Lens_Modeling_Auto.plot_functions import make_cornerPlots
from Lens_Modeling_Auto.plot_functions import plot_mcmc_behaviour_alt

results_path = '<path to modeling results>'

chainList_path = results_path + '/chain_lists'
chainList_init_path = results_path + '/chain_lists_init'
chainPlot_path = results_path + '/chainPlot_results'
chainPlot_init_path = results_path + '/chainPlot_init_results'
cornerPlot_path = results_path + '/cornerPlot_results'

if not exists(results_path + '/chainPlot_results'):
    os.mkdir(results_path + '/chainPlot_results')
if not exists(results_path + '/chainPlot_init_results'):
    os.mkdir(results_path + '/chainPlot_init_results')
if not exists(results_path + '/cornerPlot_results'):
    os.mkdir(results_path + '/cornerPlot_results')
    
chain_files = [f for f in listdir(chainList_path) if isfile('/'.join([chainList_path,f]))]
chain_files = sorted(chain_files, key=lambda k: int(re.findall('\d+', k)[0]))
chain_files_init = [f for f in listdir(chainList_init_path) if isfile('/'.join([chainList_init_path,f]))]
chain_files_init = sorted(chain_files_init, key=lambda k: int(re.findall('\d+', k)[0]))

for i,x in enumerate(chain_files):
    num = re.findall('\d+', x)[0]
    ID = re.findall('\d+', x)[1]
    print('\n')
    print('Image: {}'.format(num))
    print('ID: {}'.format(ID))
    
    
    # Load data (deserialize)
    with open(chainList_path + '/' + x , 'rb') as handle:
        chain_list = pickle.load(handle)
        
    with open(chainList_init_path + '/' + chain_files_init[i] , 'rb') as handle:
        chain_list_init = pickle.load(handle)
        
#     print(chain_list[0][1:])
    print('Making Chain Plots (initial PSO)')
    make_chainPlots(chain_list_init, chainPlot_init_path, num, ID)
    print('Making Chain Plots (final PSO)')
    make_chainPlots(chain_list, chainPlot_path, num, ID)
    
    print('\n')
    print('Making corner plots')
    make_cornerPlots(chain_list,cornerPlot_path,num, ID,step=1)
    print('\n')
#     sampler_type, samples_mcmc, param_mcmc, dist_mcmc  = chain_list[1]
#     print(samples_mcmc)
    
    del chain_list
