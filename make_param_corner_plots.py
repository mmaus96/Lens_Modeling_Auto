import sys
if sys.platform[:5] == 'linux':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import astropy.io.fits as pyfits
import astropy.io.ascii as ascii
import scipy
import pandas as pd
from scipy.ndimage.filters import gaussian_filter as gauss1D
from scipy import optimize
from pylab import figure, cm
from matplotlib.colors import LogNorm
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Util.param_util import ellipticity2phi_q
from lenstronomy.Util.param_util import shear_cartesian2polar
from os import walk
from os import listdir
from os.path import isfile, join, exists
from Lens_Modeling_Auto.auto_modeling_functions import df_2_dict
from math import ceil
from copy import deepcopy
import corner
import seaborn as sns
from scipy.stats import norm

#### Create dataframe from csv file(s) ####

path = '<parent folder>/'
csv_path = path + '<modeling results>/'
results_path = path + '<modeling results>'

# path = '/lens_candidates/Group1/'
# csv_path = path + 'SIE_lens/results_Ap30/'
# results_path = path + 'SIE_lens/results_Ap30'



if not exists(results_path):
    os.mkdir(results_path)
    
lens_model_list = ['SIE','SHEAR']
# lens_model_list = ['PEMD','SHEAR']
source_model_list = ['SERSIC_ELLIPSE']
lens_light_model_list = ['SERSIC_ELLIPSE']
# source_model_list = []
# lens_light_model_list = []
# band_list = ['g','r','i']
band_list = ['r']

df = pd.read_csv(csv_path + 'full_results_sorted.csv',delimiter =',')

dict_results = df_2_dict(df,band_list,lens_model_list,source_model_list,lens_light_model_list)

vmin, vmax = 1.0, 2.0
cmap = plt.cm.coolwarm

######## Lens plots #################

print('Making lens plots')
var_names = ['reduced chi^2','SIE_lens.theta_E','SIE_lens.center_x','SIE_lens.center_y']
# var_names = ['reduced chi^2','SIE_lens.theta_E','SIE_lens.e1','SIE_lens.e2',
#              'SIE_lens.center_x','SIE_lens.center_y','SHEAR_lens.gamma1','SHEAR_lens.gamma2']
df_plots = df[var_names]
df_plots['SIE_lens.q'] = dict_results['lens']['SIE']['q']
df_plots['SIE_lens.phi'] = dict_results['lens']['SIE']['phi']
df_plots['SHEAR_lens.gamma'] = dict_results['lens']['SHEAR']['gamma']
df_plots['SHEAR_lens.theta'] = dict_results['lens']['SHEAR']['theta']
var_names.extend(['SIE_lens.q','SIE_lens.phi','SHEAR_lens.gamma','SHEAR_lens.theta'])
df_cut = df_plots#[df_plots['reduced chi^2'] <= 3.0]
headers = ['theta_E','center_x','center_y','ell-q','ell-phi','shear-gamma','shear-theta']
# headers = ['theta_E','e1', 'e2', 'center_x','center_y','gamma1','gamma2']
# df_plots
print(var_names)
print(headers)

df_lens = deepcopy(df_plots)
var_names_lens = deepcopy(var_names)
headers_lens = deepcopy(headers)

sns.set(font_scale=2.5)
# headers = list(df_cut)
g = sns.PairGrid(df_cut,palette = 'seismic',diag_sharey=False, corner=True,height=5,vars=var_names[1:])

def facet_scatter(x, y, c, **kwargs):
    kwargs.pop("color")
    plt.scatter(x, y, c=c, **kwargs)

# vmin, vmax = 1.0, 3.0
# cmap = plt.cm.seismic

norm=plt.Normalize(vmin=vmin, vmax=vmax)
g.map_diag(plt.hist)
# g.map_lower(sns.scatterplot,s = 100,hue = df_plots['reduced chi^2'])

g.map_lower(facet_scatter,c = df_cut['reduced chi^2'],s=200, 
            vmin=vmin, vmax=vmax,norm=norm, cmap=cmap)

# Make space for the colorbar
g.fig.subplots_adjust(right=.92)

# Define a new Axes where the colorbar will go
cax = g.fig.add_axes([.94, .25, .02, .6])

# Get a mappable object with the same colormap as the data
points = plt.scatter([], [], c=[], vmin=vmin, vmax=vmax, cmap=cmap)

# Draw the colorbar
g.fig.colorbar(points, cax=cax)
# g.map_diag.set_titles('{col_name}')

# g.axes[0,0].set_title('reduced_chi^2')
for i,x in enumerate(headers):
    g.axes[i,i].set_title(x)
    g.axes[-1,i].set_xlabel(x)
for i,x in enumerate(headers[1:]):
    g.axes[i+1,0].set_ylabel(x)
    
g.savefig(results_path + '/lens_params_corner_plots.png',dpi=100)
plt.cla() 
plt.clf() 
plt.close()

del g,df_cut,headers

########### Source Plots ##################

var_names = ['reduced chi^2',
             'r Band: SERSIC_ELLIPSE_source.R_sersic',
             'r Band: SERSIC_ELLIPSE_source.n_sersic',
             'r Band: SERSIC_ELLIPSE_source.center_x',
             'r Band: SERSIC_ELLIPSE_source.center_y']
df_cut = df[var_names]
df_cut['r Band: SERSIC_ELLIPSE_source.q'] = dict_results['source']['r Band: SERSIC_ELLIPSE']['q']
df_cut['r Band: SERSIC_ELLIPSE_source.phi'] = dict_results['source']['r Band: SERSIC_ELLIPSE']['phi']
var_names.extend(['r Band: SERSIC_ELLIPSE_source.q','r Band: SERSIC_ELLIPSE_source.phi'])
headers = ['R_sersic','n_sersic','center_x','center_y','q','phi']
# df_cut = df_cut[df_cut['reduced chi^2'] <= 2.0]
print('\n')
print('Making source plots')
print(var_names)
print(headers)

df_source = deepcopy(df_cut)
var_names_source = deepcopy(var_names[1:])
headers_source = deepcopy(headers)

sns.set(font_scale=2.5)
# headers = list(df_cut)
g = sns.PairGrid(df_cut,palette = 'seismic',diag_sharey=False, corner=True,height=5,vars=var_names[1:])


# vmin, vmax = 1.0, 3.0
# cmap = plt.cm.seismic

norm=plt.Normalize(vmin=vmin, vmax=vmax)
g.map_diag(plt.hist)
# g.map_lower(sns.scatterplot,s = 100,hue = df_plots['reduced chi^2'])

g.map_lower(facet_scatter,c = df_cut['reduced chi^2'],s=200, 
            vmin=vmin, vmax=vmax,norm=norm, cmap=cmap)

# Make space for the colorbar
g.fig.subplots_adjust(right=.92)

# Define a new Axes where the colorbar will go
cax = g.fig.add_axes([.94, .25, .02, .6])

# Get a mappable object with the same colormap as the data
points = plt.scatter([], [], c=[], vmin=vmin, vmax=vmax, cmap=cmap)

# Draw the colorbar
g.fig.colorbar(points, cax=cax)
# g.map_diag.set_titles('{col_name}')

# g.axes[0,0].set_title('reduced_chi^2')
for i,x in enumerate(headers):
    g.axes[i,i].set_title(x)
    g.axes[-1,i].set_xlabel(x)
for i,x in enumerate(headers[1:]):
    g.axes[i+1,0].set_ylabel(x)
    
g.savefig(results_path + '/source_params_corner_plots.png',dpi=100)

plt.cla() 
plt.clf() 
plt.close()

del g,df_cut,headers

########### Lens light Plots ##################

var_names = ['reduced chi^2',
             'r Band: SERSIC_ELLIPSE_lens_light.R_sersic',
             'r Band: SERSIC_ELLIPSE_lens_light.n_sersic',
             'r Band: SERSIC_ELLIPSE_lens_light.center_x',
             'r Band: SERSIC_ELLIPSE_lens_light.center_y']
df_cut = df[var_names]
df_cut['r Band: SERSIC_ELLIPSE_lens_light.q'] = dict_results['lens_light']['r Band: SERSIC_ELLIPSE']['q']
df_cut['r Band: SERSIC_ELLIPSE_lens_light.phi'] = dict_results['lens_light']['r Band: SERSIC_ELLIPSE']['phi']
var_names.extend(['r Band: SERSIC_ELLIPSE_lens_light.q','r Band: SERSIC_ELLIPSE_lens_light.phi'])
headers = ['R_sersic','n_sersic','center_x','center_y','q','phi']
# df_cut = df_cut[df_cut['reduced chi^2'] <= 2.0]

print('\n')
print('Making lens_light plots')

df_lens_light = deepcopy(df_cut)
var_names_lens_light = deepcopy(var_names[1:])
headers_lens_light = deepcopy(headers)

sns.set(font_scale=2.5)
# headers = list(df_cut)
g = sns.PairGrid(df_cut,palette = 'seismic',diag_sharey=False, corner=True,height=5,vars=var_names[1:])




norm=plt.Normalize(vmin=vmin, vmax=vmax)
g.map_diag(plt.hist)
# g.map_lower(sns.scatterplot,s = 100,hue = df_plots['reduced chi^2'])

g.map_lower(facet_scatter,c = df_cut['reduced chi^2'],s=200, 
            vmin=vmin, vmax=vmax,norm=norm, cmap=cmap)

# Make space for the colorbar
g.fig.subplots_adjust(right=.92)

# Define a new Axes where the colorbar will go
cax = g.fig.add_axes([.94, .25, .02, .6])

# Get a mappable object with the same colormap as the data
points = plt.scatter([], [], c=[], vmin=vmin, vmax=vmax, cmap=cmap)

# Draw the colorbar
g.fig.colorbar(points, cax=cax)
# g.map_diag.set_titles('{col_name}')

# g.axes[0,0].set_title('reduced_chi^2')
for i,x in enumerate(headers):
    g.axes[i,i].set_title(x)
    g.axes[-1,i].set_xlabel(x)
for i,x in enumerate(headers[1:]):
    g.axes[i+1,0].set_ylabel(x)
    
g.savefig(results_path + '/lens_light_params_corner_plots.png',dpi=100)

plt.cla() 
plt.clf() 
plt.close()

del g,df_cut,headers

vars_all = var_names_lens + var_names_lens_light + var_names_source
print('\n')
print('Making Full plots')
print(vars_all)
# print(headers)
df_big = df_lens.join(df_lens_light[var_names_lens_light])
df_big = df_big.join(df_source[var_names_source])
# df.join(other.set_index('key'), on='key')
headers = ['lens.$R_E$','lens.$c_x$','lens.$c_y$','lens.$q_m$','lens.$\phi_m$','lens.$\gamma_{ext}$','lens.$\phi_{ext}$']
headers.extend(['lens light.$R_{eff}$','lens light.$n_{s}$','lens light.$c_{x}$',
                'lens light.$c_{y}$','lens light.$q_{l}$','lens light.$\phi_{l}$'])

headers.extend(['source.$R_{eff}$','source.$n_{s}$','source.$c_{x}$',
                'source.$c_{y}$','source.$q_{l}$','source.$\phi_{l}$'])
print(headers)





g = sns.PairGrid(df_big,palette = 'seismic',diag_sharey=False, corner=True,height=5,vars=vars_all[1:])




norm=plt.Normalize(vmin=vmin, vmax=vmax)
g.map_diag(plt.hist)
# g.map_lower(sns.scatterplot,s = 100,hue = df_plots['reduced chi^2'])

g.map_lower(facet_scatter,c = df_big['reduced chi^2'],s=200, 
            vmin=vmin, vmax=vmax,norm=norm, cmap=cmap)

# Make space for the colorbar
g.fig.subplots_adjust(right=.92)

# Define a new Axes where the colorbar will go
cax = g.fig.add_axes([.94, .25, .02, .6])

# Get a mappable object with the same colormap as the data
points = plt.scatter([], [], c=[], vmin=vmin, vmax=vmax, cmap=cmap)

# Draw the colorbar
g.fig.colorbar(points, cax=cax)
# g.map_diag.set_titles('{col_name}')

# g.axes[0,0].set_title('reduced_chi^2')
for i,x in enumerate(headers):
    g.axes[i,i].set_title(x)
    g.axes[-1,i].set_xlabel(x)
for i,x in enumerate(headers[1:]):
    g.axes[i+1,0].set_ylabel(x)
    
g.savefig(results_path + '/All_params_corner_plots.png',dpi=100)

plt.cla() 
plt.clf() 
plt.close()

# del g,df_cut,headers




