from lenstronomy.Plots import chain_plot
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Plots.chain_plot import plot_mcmc_behaviour
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from os.path import exists



############################ modelPlots #################################

modelPlot = ModelPlot(multi_band_list, kwargs_model, kwargs_result, arrow_size=0.02, cmap_string="gist_heat",likelihood_mask_list= mask_list)

n_data = modelPlot._imageModel.num_data_evaluate
logL = modelPlot._imageModel.likelihood_data_given_model(source_marg=False, linear_prior=None, **kwargs_result)
red_X_squared = np.abs(logL * 2.0 / n_data)

for l,b in enumerate(band_list):
    f, axes = plt.subplots(4, 3, figsize=(20,20), sharex=False, sharey=False)
    
    band_path = modelPlot_path + '/' + b
    
    if not exists(band_path):
        os.mkdir(band_path)

    modelPlot.data_plot(ax=axes[0,0],band_index=l)
    modelPlot.model_plot(ax=axes[0,1],band_index=l)
    modelPlot.normalized_residual_plot(ax=axes[0,2], v_min=-6, v_max=6,band_index=l)
    modelPlot.source_plot(ax=axes[1, 0], deltaPix_source=0.01, numPix=100,band_index=l)
    modelPlot.convergence_plot(ax=axes[1, 1], v_max=1,band_index=l)
    modelPlot.magnification_plot(ax=axes[1, 2],band_index=l)
    modelPlot.decomposition_plot(ax=axes[2,0], text='Lens light', lens_light_add=True, unconvolved=True,band_index=l)
    modelPlot.decomposition_plot(ax=axes[3,0], text='Lens light convolved', lens_light_add=True,band_index=l)
    modelPlot.decomposition_plot(ax=axes[2,1], text='Source light', source_add=True, unconvolved=True,band_index=l)
    modelPlot.decomposition_plot(ax=axes[3,1], text='Source light convolved', source_add=True,band_index=l)
    modelPlot.decomposition_plot(ax=axes[2,2], text='All components', source_add=True, lens_light_add=True, unconvolved=True,band_index=l)
    modelPlot.decomposition_plot(ax=axes[3,2], text='All components convolved', source_add=True, lens_light_add=True, point_source_add=True,band_index=l)
    f.tight_layout()
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    f.savefig(band_path + '/Image_{}-{}.png'.format(it+1,data_pairs_dicts[it]['object_ID']),dpi = 200)
    f.clear()
    plt.close(f)


############################ chainPlots #################################

####PSO chain plots
chain_pso = chain_list[0]
chain, param_list = chain_pso[1:]
X2_list, pos_list, vel_list = chain
f, axes = plt.subplots(2, 2, figsize=(16, 16))

ax = axes[0,0]
ax.plot(np.log10(-np.array(X2_list)))
ax.set_title('-logL')

ax = axes[0,1]
pos = np.array(pos_list)
vel = np.array(vel_list)
n_iter = len(pos)
for i in range(0, len(pos[0])):
    ax.plot((pos[:, i]-pos[n_iter-1, i]) / (pos[n_iter-1, i] + 1), label=param_list[i])
ax.set_title('particle position')
ax.legend()

ax = axes[1,0]
for i in range(0,len(vel[0])):
    ax.plot(vel[:, i] / (pos[n_iter-1, i] + 1), label=param_list[i])
ax.set_title('param velocity')
ax.legend()

### MCMC chain plot
chain_mcmc = chain_list[1]
samples, param, dist = chain_mcmc[1:]
ax = axes[1,1]
plot_mcmc_behaviour(ax, samples, param, dist, num_average=100)

f.savefig(chainPlot_path + '/Image_{}-{}.png'.format(it+1,data_pairs_dicts[it]['object_ID']),dpi = 200)
f.clear()
plt.close(f)



############################ cornerPlots #################################

sampler_type, samples_mcmc, param_mcmc, dist_mcmc  = chain_list[1]

print("number of non-linear parameters in the MCMC process: ", len(param_mcmc))
print("parameters in order: ", param_mcmc)
print("number of evaluations in the MCMC process: ", np.shape(samples_mcmc)[0])
import corner
if not samples_mcmc == []:
    n, num_param = np.shape(samples_mcmc)
    plot = corner.corner(samples_mcmc[:,:], labels=param_mcmc[:], show_titles=True)
    plot.savefig(cornerPlot_path + '/Image_{}-{}.png'.format(it+1,data_pairs_dicts[it]['object_ID']),dpi=200)
    plot.clear()
    plt.close(plot)

    
del chain_list, f, axes, plot, param_mcmc, modelPlot, sampler_type, samples_mcmc, dist_mcmc
del chain_mcmc, samples, param, dist, chain_pso, chain, param_list, X2_list, pos_list, vel_list