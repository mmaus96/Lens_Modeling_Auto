from lenstronomy.Plots import chain_plot
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Analysis.image_reconstruction import ModelBand
from lenstronomy.Plots.chain_plot import plot_mcmc_behaviour
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Plots import plot_util
from mpl_toolkits.axes_grid1 import make_axes_locatable
import lenstronomy.Util.util as util
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from os.path import exists
from copy import deepcopy
import pickle
import math
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv


############################ plot lines and annotations #################################

def text_description(ax, d, text, color='w', backgroundcolor='k',
                     flipped=False, font_size=15):
    c_vertical = 1/15. #+ font_size / d / 10.**2
    c_horizontal = 1./30
#     if flipped:
#         ax.text(d - d * c_horizontal, d - d * c_vertical, text, color=color,
#                 fontsize=font_size,backgroundcolor=backgroundcolor)
#     else:
#         ax.text(d * c_horizontal, d - d * c_vertical, text, color=color, fontsize=font_size,backgroundcolor=backgroundcolor)
    if flipped:
        ax.text(0.1,0.9, text, color=color,
                fontsize=font_size,transform=ax.transAxes,backgroundcolor=backgroundcolor)
    else:
        ax.text(0.01,0.99, text, horizontalalignment='left',verticalalignment='top',
                color=color, bbox={'facecolor':backgroundcolor, 'edgecolor':'none', 'pad':1},
                fontsize=font_size,transform=ax.transAxes,backgroundcolor=backgroundcolor)
        
def coordinate_arrows(ax, d, coords, color='w', font_size=15, arrow_size=0.05):
    d0 = d / 8.
    p0 = d / 15.
    pt = d / 9.
    deltaPix = coords.pixel_width
    ra0, dec0 = coords.map_pix2coord((d - d0) / deltaPix, d0 / deltaPix)
    xx_, yy_ = coords.map_coord2pix(ra0, dec0)
    xx_ra, yy_ra = coords.map_coord2pix(ra0 + p0, dec0)
    xx_dec, yy_dec = coords.map_coord2pix(ra0, dec0 + p0)
    xx_ra_t, yy_ra_t = coords.map_coord2pix(ra0 + pt, dec0)
    xx_dec_t, yy_dec_t = coords.map_coord2pix(ra0, dec0 + pt)

    ax.arrow(xx_ * deltaPix, yy_ * deltaPix, (xx_ra - xx_) * deltaPix, (yy_ra - yy_) * deltaPix,
             head_width=arrow_size * d, head_length=arrow_size * d, fc=color, ec=color, linewidth=0.5,rasterized=True)
    ax.text(xx_ra_t * deltaPix, yy_ra_t * deltaPix, "E", color=color,weight='normal', fontsize=font_size, ha='center',rasterized=True)
    ax.arrow(xx_ * deltaPix, yy_ * deltaPix, (xx_dec - xx_) * deltaPix, (yy_dec - yy_) * deltaPix,
             head_width=arrow_size * d, head_length=arrow_size * d, fc
             =color, ec=color, linewidth=0.5,rasterized=True)
    ax.text(xx_dec_t * deltaPix, yy_dec_t * deltaPix, "N", color=color,weight='normal', fontsize=font_size, ha='center',rasterized=True)

def scale_bar(ax, d, dist=1., text='1"', color='w', font_size=15, flipped=False,label=True):
    if flipped:
        p0 = d - d / 15. - dist
        p1 = d / 15.
        ax.plot([p0, p0 + dist], [p1, p1], linewidth=1, color=color,rasterized=True)
        if label:
            ax.text(p0 + dist / 2., p1 + 0.01 * d, text,rasterized=True, fontsize=font_size,
                color=color, ha='center',weight='normal')
    else:
        p0 = d / 15.
        ax.plot([p0, p0 + dist], [p0, p0], linewidth=1, color=color,rasterized=True)
        if label:
            ax.text(p0 + dist / 2., p0 + 0.01 * d, text, fontsize=font_size,weight='normal',rasterized=True, color=color, ha='center')


def plot_line_set(ax, coords, line_set_list_x, line_set_list_y, origin=None, color='g', flipped_x=False):
    """
    plotting a line set on a matplotlib instance where the coordinates are defined in pixel units with the lower left
    corner (defined as origin) is by default (0, 0). The coordinates are moved by 0.5 pixels to be placed in the center
    of the pixel in accordance with the matplotlib.matshow() routine.

    :param ax: matplotlib.axis instance
    :param coords: Coordinates() class instance
    :param origin: [x0, y0], lower left pixel coordinate in the frame of the pixels
    :param line_set_list_x: numpy arrays corresponding of different disconnected regions of the line
     (e.g. caustic or critical curve)
    :param line_set_list_y: numpy arrays corresponding of different disconnected regions of the line
     (e.g. caustic or critical curve)
    :param color: string with matplotlib color
    :param flipped_x: bool, if True, flips x-axis
    :return: plot with line sets on matplotlib axis in pixel coordinates
    """
    if origin is None:
        origin = [0, 0]
    pixel_width = coords.pixel_width
    pixel_width_x = pixel_width
    if flipped_x:
        pixel_width_x = -pixel_width
    x_c, y_c = coords.map_coord2pix(line_set_list_x, line_set_list_y)
    ax.plot((x_c + 0.5) * pixel_width_x + origin[0], (y_c + 0.5) * pixel_width + origin[1],
            ',',rasterized=True, color=color)
    return ax

############################ Plot data  #################################


def plot_data(plot_class,ax,data,cut_val = -5, v_min=None, v_max=None, text=None,
                  font_size=15, colorbar_label=r'log$_{10}$ flux',cmap = 'cubehelix',
              scale_bar_label = True,cb_tick_size=15,**kwargs):
        """

        :param ax:
        :return:
        """
        
        log_data = np.log10(data)
        log_data[np.isnan(log_data)] = cut_val
#         v_min = max(np.min(log_data), cut_val)
#         v_max = min(np.max(log_data), 10)
        if v_min is None:
            v_min = max(np.min(log_data), cut_val)
        if v_max is None:
            v_max = min(np.max(log_data), 10)
            
        
#         im = ax.matshow(data,origin='lower', norm = SymLogNorm(linthresh,vmin=v_min,vmax = v_max),
#                     extent=[0, plot_class._frame_size, 0, plot_class._frame_size], cmap=plot_class._cmap)
        im = ax.matshow(log_data,origin='lower', vmin=v_min,vmax = v_max,rasterized=True,    
                    extent=[0, plot_class._frame_size, 0, plot_class._frame_size], cmap=plot_class._cmap)
        
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
#         ax.set_rasterized(True)

        scale_bar(ax, plot_class._frame_size, dist=1, text='1"', font_size=font_size,label=scale_bar_label)
        text_description(ax, plot_class._frame_size, text=text, color="w",
                         backgroundcolor='k', font_size=font_size)

        if 'no_arrow' not in kwargs or not kwargs['no_arrow']:
            coordinate_arrows(ax, plot_class._frame_size, plot_class._coords, color='w',
                              arrow_size=plot_class._arrow_size, font_size=font_size)
        

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad='2%')
        
        cb = plt.colorbar(im, cax=cax, orientation='horizontal')
        if colorbar_label != None:
            cb.set_label(colorbar_label, fontsize=font_size)
        cb.ax.tick_params(labelsize=cb_tick_size)
        cax.xaxis.set_ticks_position("top")
        cb.ax.xaxis.set_label_position('top')
        cb.ax.set_rasterized(True)
#         cb.set_ticks(tick_locations)
#         cb.set_ticklabels(['{}'.format(x) for x in tick_locations])
        return ax,cb


############################ Plot source  #################################

def source_plot(plot_class, ax, numPix, deltaPix_source, center=None, v_min=None,
                    v_max=None, with_caustics=False, caustic_color='yellow',
                    font_size=15, plot_scale='log',
                    scale_size=0.1,
                    text="Reconstructed source",cb_tick_size=15,
                    colorbar_label=r'log$_{10}$ flux', point_source_position=True,scale_bar_label = True,
                    **kwargs):
        """

        :param ax:
        :param numPix:
        :param deltaPix_source:
        :param center: [center_x, center_y], if specified, uses this as the center
        :param v_min:
        :param v_max:
        :param with_caustics:
        :param caustic_color:
        :param font_size:
        :param plot_scale: string, log or linear, scale of surface brightness plot
        :param kwargs:
        :return:
        """
        if v_min is None:
            v_min = plot_class._v_min_default
        if v_max is None:
            v_max = plot_class._v_max_default
        d_s = numPix * deltaPix_source
        source, coords_source = plot_class.source(numPix, deltaPix_source, center=center)
        if plot_scale == 'log':
            source[source < 10**(v_min)] = 10**(v_min) # to remove weird shadow in plot
            source_scale = np.log10(source)
        elif plot_scale == 'linear':
            source_scale = source
        else:
            raise ValueError('variable plot_scale needs to be "log" or "linear", not %s.' % plot_scale)
        im = ax.matshow(source_scale, origin='lower', extent=[0, d_s, 0, d_s],rasterized=True,
                        cmap=plot_class._cmap, vmin=v_min, vmax=v_max)  # source
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
#         ax.set_rasterized(True)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad='2%')
        
        cb = plt.colorbar(im, cax=cax,orientation='horizontal')
        
        if colorbar_label != None:
            cb.set_label(colorbar_label, fontsize=font_size)
        cb.ax.tick_params(labelsize=cb_tick_size)
        cax.xaxis.set_ticks_position("top")
        cb.ax.xaxis.set_label_position('top')
        cb.ax.set_rasterized(True)
        text_description(ax, d_s, text=text, color="w", backgroundcolor='k',
                         flipped=False, font_size=font_size)
        if with_caustics is True:
            ra_caustic_list, dec_caustic_list = plot_class._caustics()
            plot_line_set(ax, coords_source, ra_caustic_list,
                          dec_caustic_list, color=caustic_color)
            scale_bar(ax, d_s, dist=scale_size, text='{:.1f}"'.format(scale_size),
                  color='w',
                  flipped=False,
                  font_size=font_size,
                  label = scale_bar_label
                               )
#             text_description(ax, d_s, text=text, color="w", backgroundcolor='k',
#                          flipped=False, font_size=font_size)
        if 'no_arrow' not in kwargs or not kwargs['no_arrow']:
            plot_util.coordinate_arrows(ax, plot_class._frame_size, plot_class._coords, color='w',
                              arrow_size=plot_class._arrow_size, font_size=font_size)
#             text_description(ax, d_s, text=text, color="w", backgroundcolor='k',
#                          flipped=False, font_size=font_size)
        if point_source_position is True:
            ra_source, dec_source = plot_class._bandmodel.PointSource.source_position(plot_class._kwargs_ps_partial, plot_class._kwargs_lens_partial)
            plot_util.source_position_plot(ax, coords_source, ra_source, dec_source)
        return ax,cb
    


############################ Subtract from data  #################################

def subtract_from_data_plot(plot_class, ax, text='Subtracted',cut_val = -5, v_min=None,
                                v_max=None, point_source_add=False,
                                source_add=False, lens_light_add=False,
                                font_size=15,cb_tick_size=15
                                ):
        model = plot_class._bandmodel.image(plot_class._kwargs_lens_partial, plot_class._kwargs_source_partial, plot_class._kwargs_lens_light_partial,
                                          plot_class._kwargs_ps_partial, unconvolved=False, source_add=source_add,
                                          lens_light_add=lens_light_add, point_source_add=point_source_add)
        
        data_sub = plot_class._data - model
        log_data = np.log10(data_sub)
        log_data[np.isnan(log_data)] = cut_val
        if v_min is None:
            v_min = max(np.min(log_data), cut_val)
        if v_max is None:
            v_max = min(np.max(log_data), 10)
        
            
        
        im = ax.matshow(log_data, origin='lower', vmin=v_min, vmax=v_max,rasterized=True,
                        extent=[0, plot_class._frame_size, 0, plot_class._frame_size], cmap=plot_class._cmap)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
#         ax.set_rasterized(True)
        plot_util.scale_bar(ax, plot_class._frame_size, dist=1, text='1"', font_size=font_size)
        plot_util.text_description(ax, plot_class._frame_size, text=text, color="w",
                         backgroundcolor='k', font_size=font_size)
        plot_util.coordinate_arrows(ax, plot_class._frame_size, plot_class._coords,
                          arrow_size=plot_class._arrow_size, font_size=font_size)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad='2%')
        cax.xaxis.set_ticks_position("top")
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(r'log$_{10}$ flux', fontsize=font_size)
        cb.ax.tick_params(labelsize=cb_tick_size)
        cb.ax.set_rasterized(True)
        return ax,cb
    
############################ Residuals #################################
    
def normalized_residual_plot(plot_class, ax, v_min=-6, v_max=6, font_size=15, text="Normalized Residuals",
                                 colorbar_label=r'(f${}_{\rm model}$ - f${}_{\rm data}$)/$\sigma$',
                                 no_arrow=False,cb_tick_size=15,scale_bar_label=True, **kwargs):
        """

        :param ax:
        :param v_min:
        :param v_max:
        :param kwargs: kwargs to send to matplotlib.pyplot.matshow()
        :return:
        """
        if not 'cmap' in kwargs:
            kwargs['cmap'] = 'bwr'
        im = ax.matshow(plot_class._norm_residuals, vmin=v_min, vmax=v_max,rasterized=True,
                        extent=[0, plot_class._frame_size, 0, plot_class._frame_size], origin='lower',
                        **kwargs)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
#         ax.set_rasterized(True)
        scale_bar(ax, plot_class._frame_size, dist=1, text='1"', color='k',
                  font_size=font_size,label=scale_bar_label)
        text_description(ax, plot_class._frame_size, text=text, color="k",
                         backgroundcolor='w', font_size=font_size)
        if not no_arrow:
            plot_util.coordinate_arrows(ax, plot_class._frame_size, plot_class._coords, color='w',
                              arrow_size=plot_class._arrow_size, font_size=font_size)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad='2%')
        
        cb = plt.colorbar(im, cax=cax,orientation='horizontal')
        if colorbar_label != None:
            cb.set_label(colorbar_label, fontsize=font_size)
        cb.ax.tick_params(labelsize=cb_tick_size)
        cax.xaxis.set_ticks_position("top")
        cb.ax.xaxis.set_label_position('top')

        cb.ax.set_rasterized(True)
        return ax,cb
    
    
############################ magnification #################################    
    
def magnification_plot(plot_class, ax, v_min=-10, v_max=10,
                           image_name_list=None, font_size=15, no_arrow=False,
                           text="Magnification model",
                           colorbar_label=r"$\det\ (\mathsf{A}^{-1})$",cb_tick_size=15,
                           **kwargs):
        """

        :param ax: matplotib axis instance
        :param v_min: minimum range of plotting
        :param v_max: maximum range of plotting
        :param kwargs: kwargs to send to matplotlib.pyplot.matshow()
        :return:
        """
        if not 'cmap' in kwargs:
            kwargs['cmap'] = plot_class._cmap
        if not 'alpha' in kwargs:
            kwargs['alpha'] = 0.5
        mag_result = util.array2image(plot_class._lensModel.magnification(plot_class._x_grid, plot_class._y_grid, plot_class._kwargs_lens_partial))
        im = ax.matshow(mag_result, origin='lower', extent=[0, plot_class._frame_size, 0, plot_class._frame_size],
                        rasterized=True,vmin=v_min, vmax=v_max, **kwargs)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
#         ax.set_rasterized(True)
        plot_util.scale_bar(ax, plot_class._frame_size, dist=1, text='1"', color='k', font_size=font_size)
        if not no_arrow:
            coordinate_arrows(ax, plot_class._frame_size, plot_class._coords, color='k', arrow_size=plot_class._arrow_size,
                                        font_size=font_size)
        text_description(ax, plot_class._frame_size, text=text, color="k",
                         backgroundcolor='w', font_size=font_size)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad='2%')
        cax.xaxis.set_ticks_position("top")
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(colorbar_label, fontsize=font_size)
        cb.ax.tick_params(labelsize=cb_tick_size)
        
        cb.ax.set_rasterized(True)
        ra_image, dec_image = plot_class._bandmodel.PointSource.image_position(plot_class._kwargs_ps_partial, plot_class._kwargs_lens_partial)
        plot_util.image_position_plot(ax, plot_class._coords, ra_image, dec_image, color='k', image_name_list=image_name_list)
        return ax,cb    
    
############################ magnification ################################# 
def convergence_plot(plot_class, ax, text='Convergence', v_min=None, v_max=None,
                         font_size=15, colorbar_label=r'$\log_{10}\ \kappa$',cb_tick_size=15,
                         scale_bar_label=True,**kwargs):
        """

        :param x_grid:
        :param y_grid:
        :param kwargs_lens:
        :param kwargs_else:
        :return:
        """
        if not 'cmap' in kwargs:
            kwargs['cmap'] = plot_class._cmap

        kappa_result = util.array2image(plot_class._lensModel.kappa(plot_class._x_grid, plot_class._y_grid, plot_class._kwargs_lens_partial))
        im = ax.matshow(np.log10(kappa_result), origin='lower',rasterized=True,
                        extent=[0, plot_class._frame_size, 0, plot_class._frame_size],
                        cmap=kwargs['cmap'], vmin=v_min, vmax=v_max)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
#         ax.set_rasterized(True)
        scale_bar(ax, plot_class._frame_size, dist=1, text='1"', color='w', 
                  font_size=font_size,label=scale_bar_label)
        text_description(ax, plot_class._frame_size, text=text,
                         color="w", backgroundcolor='k', flipped=False,
                         font_size=font_size)
        if 'no_arrow' not in kwargs or not kwargs['no_arrow']:
            coordinate_arrows(ax, plot_class._frame_size, plot_class._coords, color='w',
                              arrow_size=plot_class._arrow_size, font_size=font_size)
            
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad='2%')
        cax.xaxis.set_ticks_position("top")
        cb = plt.colorbar(im, cax=cax,orientation='horizontal')
        if colorbar_label != None:
            cb.set_label(colorbar_label, fontsize=font_size)
        cb.ax.tick_params(labelsize=cb_tick_size)
        cax.xaxis.set_ticks_position("top")
        cb.ax.xaxis.set_label_position('top')
        
        cb.ax.set_rasterized(True)

        return ax,cb
    



############################ modelPlots #################################

def make_modelPlots(multi_band_list,kwargs_model,kwargs_result,
                    kwargs_data,kwargs_psf, lens_info,
                    lens_model_list,source_model_list,lens_light_model_list,
                    mask_list,band_list,modelPlot_path,num, object_ID):
    
    modelPlot = ModelPlot(multi_band_list, kwargs_model, kwargs_result, arrow_size=0.02, 
                          cmap_string="gist_heat",likelihood_mask_list= mask_list)
    
    lens_class = LensModel(lens_model_list=lens_model_list)
    source_class = LightModel(light_model_list = source_model_list)
    lens_light_class = LightModel(light_model_list = lens_light_model_list)
    
    n_data = modelPlot._imageModel.num_data_evaluate
    logL = modelPlot._imageModel.likelihood_data_given_model(source_marg=False, linear_prior=None, **kwargs_result)
    red_X_squared = np.abs(logL * 2.0 / n_data)
    
    model, error_map, cov_param, param = modelPlot._imageModel.image_linear_solve(inv_bool=True, **kwargs_result)
    
    for l,b in enumerate(band_list):
        f, axes = plt.subplots(4, 3, figsize=(20,20), sharex=False, sharey=False)

        band_path = modelPlot_path + '/' + b

        if not exists(band_path):
            os.mkdir(band_path)
            
        m=len(source_model_list)
        result_source = []    
        for i in range(len(source_model_list)): 
            result_source.append(kwargs_result['kwargs_source'][m*l + i])
        

        im_data = ImageData(**kwargs_data[l])
        psf_data = PSF(**kwargs_psf[l])
        im_sim = ImageModel(im_data,psf_data,lens_model_class=lens_class, 
                        source_model_class=source_class, lens_light_model_class= None)
        
#         surfaceBrightness = im_sim.source_surface_brightness(kwargs_result['kwargs_source'][l],
#                                                                  kwargs_lens=kwargs_result['kwargs_lens'],
#                                                                  de_lensed= True)
        surfaceBrightness = im_sim.source_surface_brightness(result_source,
                                                                 kwargs_lens=kwargs_result['kwargs_lens'],
                                                                 de_lensed= True)
        
        modelPlot.data_plot(ax=axes[0,0],band_index=l)
        modelPlot.model_plot(ax=axes[0,1],band_index=l)
        modelPlot.normalized_residual_plot(ax=axes[0,2], v_min=-6, v_max=6,band_index=l)
#         modelPlot.source_plot(ax=axes[1, 0], deltaPix_source=0.01, numPix=100,band_index=l)
        
        ax = axes[1, 0]
        numPix = lens_info[l]['numPix']
        deltaPix = lens_info[l]['deltaPix']
        d_s = numPix * deltaPix
        im = ax.matshow(surfaceBrightness,cmap='gist_heat' ,rasterized=True,origin='lower',extent=[0, d_s, 0, d_s])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label('surface brightness', fontsize=15)
        plot_util.text_description(ax, d_s, text="Reconstructed source", color="w", backgroundcolor='k',
                         flipped=False, font_size=15)
        scale_size = 1
        plot_util.scale_bar(ax, d_s, dist=scale_size, text='{}"'.format(scale_size),
                  color='w',
                  flipped=False,
                  font_size=15)
        
        modelPlot.convergence_plot(ax=axes[1, 1], v_max=1,band_index=l)
        modelPlot.magnification_plot(ax=axes[1, 2],band_index=l)
        modelPlot.decomposition_plot(ax=axes[2,0], text='Lens light', lens_light_add=True, unconvolved=True,band_index=l)
        modelPlot.decomposition_plot(ax=axes[3,0], text='Lens light convolved', lens_light_add=True,band_index=l)
        modelPlot.decomposition_plot(ax=axes[2,1], text='Source light', source_add=True, unconvolved=True,band_index=l)
        modelPlot.decomposition_plot(ax=axes[3,1], text='Source light convolved', source_add=True,band_index=l)
        modelPlot.decomposition_plot(ax=axes[2,2], text='All components', source_add=True, lens_light_add=True, 
                                     unconvolved=True,band_index=l)
        modelPlot.decomposition_plot(ax=axes[3,2], text='All components convolved', source_add=True, lens_light_add=True, 
                                     point_source_add=True,band_index=l)
        
        model_band = ModelBand(multi_band_list, kwargs_model, model[l], error_map[l], cov_param[l],
                                               param[l], deepcopy(kwargs_result),
                                               image_likelihood_mask_list=mask_list, band_index=l)
#         f.tight_layout()
        f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
        f.suptitle('$ID:$ {} \n $ \chi^2 $ (all): {:.4f} \n $\chi^2$({} band):{:.4f}'
                                           .format(object_ID, red_X_squared,b, 
                                           model_band._reduced_x2),fontsize=30)
        f.savefig(band_path + '/Image_{}-{}.png'.format(num,object_ID),dpi = 200)
        f.clear()
        plt.close(f)
        plt.cla()
        plt.clf()
        
    del modelPlot
        
    return red_X_squared





############################ LRG Plot results #################################


def plot_LRG_fit(plot_kwargs,band_list,path,num, object_ID):
    
    modelPlot = ModelPlot(arrow_size=0.02,cmap_string="gist_heat", **plot_kwargs)
    n_data = modelPlot._imageModel.num_data_evaluate
    logL = modelPlot._imageModel.likelihood_data_given_model(source_marg=False, linear_prior=None, **plot_kwargs['kwargs_params'])
    red_X_squared = np.abs(logL * 2.0 / n_data)
    
    nrows = len(band_list)
    f, axes = plt.subplots(nrows, 4, figsize=(4*6,nrows*5), sharex=False, sharey=False)
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    ax = axes.ravel()
    for l,b in enumerate(band_list): 
        modelPlot.data_plot(ax=ax[0+l*4],band_index=l)
        ax[0+l*4].set_title('{} Band:'.format(b), fontsize=20)
        
        modelPlot.decomposition_plot(ax=ax[1+l*4], text='Lens light', lens_light_add=True, unconvolved=True,band_index=l)
        modelPlot.model_plot(ax=ax[2+l*4],band_index=l)
        modelPlot.normalized_residual_plot(ax=ax[3+l*4], v_min=-6, v_max=6,band_index=l)
        
    f.tight_layout()
    f.savefig(path + '/Image_{}-{}.pdf'.format(num,object_ID),dpi = 100)
    f.clear()
    plt.close(f)
    plt.cla()
    plt.clf()
    
    return plot_kwargs, red_X_squared

############################ lensed source Plot results #################################


def plot_lensed_source_fit(plot_kwargs,kwargs_data, kwargs_psf,
                           band_list,lens_model_list,source_model_list,lens_light_model_list,
                           path,num, object_ID):
    
    modelPlot = ModelPlot(arrow_size=0.02,cmap_string="gist_heat", **plot_kwargs)
    lens_class = LensModel(lens_model_list=lens_model_list)
    source_class = LightModel(light_model_list = source_model_list)
    lens_light_class = LightModel(light_model_list = lens_light_model_list)
    
    kwargs_result = deepcopy(plot_kwargs['kwargs_params'])
    
    nrows = len(band_list)
    f, axes = plt.subplots(nrows, 4, figsize=(4*6,nrows*5), sharex=False, sharey=False)
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    axes = axes.ravel()
    for l,b in enumerate(band_list): 
        modelPlot.data_plot(ax=axes[0+l*4],band_index=l)
        axes[0+l*4].set_title('{} Band:'.format(b), fontsize=20)
        
        im_data = ImageData(**kwargs_data[l])
        psf_data = PSF(**kwargs_psf[l])
        im_sim = ImageModel(im_data,psf_data,lens_model_class=lens_class, 
                        source_model_class=source_class, lens_light_model_class= None)
        
        surfaceBrightness = im_sim.source_surface_brightness([kwargs_result['kwargs_source'][l]],
                                                                 kwargs_lens=kwargs_result['kwargs_lens'],
                                                                 de_lensed= True)
        ax = axes[1+l*4]
#         numPix = lens_info[l]['numPix']
#         deltaPix = lens_info[l]['deltaPix']
        deltaPix=modelPlot._band_plot_list[l]._deltaPix
        numPix=len(kwargs_data[l]['image_data'])
        
        d_s = numPix * deltaPix
        
        im = ax.matshow(surfaceBrightness,cmap='gist_heat' ,origin='lower',extent=[0, d_s, 0, d_s])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label('surface brightness', fontsize=15)
        plot_util.text_description(ax, d_s, text="Reconstructed source", color="w", backgroundcolor='k',
                         flipped=False, font_size=15)
        
        scale_size = 1
        plot_util.scale_bar(ax, d_s, dist=scale_size, text='{}"'.format(scale_size),
                  color='w',
                  flipped=False,
                  font_size=15)
        
        modelPlot.model_plot(ax=axes[2+l*4],band_index=l)
        modelPlot.normalized_residual_plot(ax=axes[3+l*4], v_min=-6, v_max=6,band_index=l)
        
    f.tight_layout()
    f.savefig(path + '/Image_{}-{}.png'.format(num,object_ID),dpi = 200)
    
    return plot_kwargs



############################ galaxyPlots ######################################
def make_galaxyPlots(multi_band_list,kwargs_model,kwargs_result,
                    kwargs_data,kwargs_psf, lens_info,
                    lens_model_list,source_model_list,lens_light_model_list,
                    mask_list,band_list,galPlot_path,num, object_ID):
    
    modelPlot = ModelPlot(multi_band_list, kwargs_model, kwargs_result, arrow_size=0.02, 
                          cmap_string="gist_heat",likelihood_mask_list= mask_list)
    
    lens_class = LensModel(lens_model_list=lens_model_list)
    source_class = LightModel(light_model_list = source_model_list)
    lens_light_class = LightModel(light_model_list = lens_light_model_list)
    
    n_data = modelPlot._imageModel.num_data_evaluate
    logL = modelPlot._imageModel.likelihood_data_given_model(source_marg=False, linear_prior=None, **kwargs_result)
    red_X_squared = np.abs(logL * 2.0 / n_data)
    
    model, error_map, cov_param, param = modelPlot._imageModel.image_linear_solve(inv_bool=True, **kwargs_result)
    
    for l,b in enumerate(band_list):
        f, axes = plt.subplots(2, 3, figsize=(20,15), sharex=False, sharey=False)
        f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.2)

        band_path = galPlot_path + '/' + b

        if not exists(band_path):
            os.mkdir(band_path)

        im_data = ImageData(**kwargs_data[l])
        psf_data = PSF(**kwargs_psf[l])
        im_sim = ImageModel(im_data,psf_data,lens_model_class=lens_class, 
                        source_model_class=source_class, lens_light_model_class= None)
        
#         surfaceBrightness = im_sim.source_surface_brightness([kwargs_result['kwargs_source'][l]],
#                                                                  kwargs_lens=kwargs_result['kwargs_lens'],
#                                                                  de_lensed= True)
        
        modelPlot.data_plot(ax=axes[0,0],band_index=l)
        modelPlot.model_plot(ax=axes[0,1],band_index=l)
        modelPlot.normalized_residual_plot(ax=axes[0,2], v_min=-6, v_max=6,band_index=l)
#         modelPlot.source_plot(ax=axes[1, 0], deltaPix_source=0.01, numPix=100,band_index=l)
        
#         ax = axes[1, 0]
#         numPix = lens_info[l]['numPix']
#         deltaPix = lens_info[l]['deltaPix']
#         d_s = numPix * deltaPix
#         im = ax.matshow(surfaceBrightness,cmap='gist_heat' ,origin='lower',extent=[0, d_s, 0, d_s])
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#         ax.autoscale(False)
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.05)
#         cb = plt.colorbar(im, cax=cax)
#         cb.set_label('surface brightness', fontsize=15)
#         plot_util.text_description(ax, d_s, text="Reconstructed source", color="w", backgroundcolor='k',
#                          flipped=False, font_size=15)
        
#         modelPlot.convergence_plot(ax=axes[1, 1], v_max=1,band_index=l)
#         modelPlot.magnification_plot(ax=axes[1, 2],band_index=l)
        modelPlot.decomposition_plot(ax=axes[1,0], text='Lens light', lens_light_add=True, unconvolved=True,band_index=l)
        modelPlot.decomposition_plot(ax=axes[1,1], text='Lens light convolved', lens_light_add=True,band_index=l)
#         modelPlot.decomposition_plot(ax=axes[2,1], text='Source light', source_add=True, unconvolved=True,band_index=l)
#         modelPlot.decomposition_plot(ax=axes[3,1], text='Source light convolved', source_add=True,band_index=l)
#         modelPlot.decomposition_plot(ax=axes[2,2], text='All components', source_add=True, lens_light_add=True, 
#                                      unconvolved=True,band_index=l)
#         modelPlot.decomposition_plot(ax=axes[3,2], text='All components convolved', source_add=True, lens_light_add=True, 
#                                      point_source_add=True,band_index=l)
        
        axes[1,2].set_axis_off()
        model_band = ModelBand(multi_band_list, kwargs_model, model[l], error_map[l], cov_param[l],
                                               param[l], deepcopy(kwargs_result),
                                               image_likelihood_mask_list=mask_list, band_index=l)
#         f.tight_layout()
        f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.0)
        f.suptitle('$ID:$ {} \n $ \chi^2 $ (all): {:.4f} \n $\chi^2$({} band):{:.4f}'
                                           .format(object_ID, red_X_squared,b, 
                                           model_band._reduced_x2),fontsize=30)
        f.savefig(band_path + '/Image_{}-{}.png'.format(num,object_ID),dpi = 200)
        f.clear()
        plt.close(f)
        plt.cla()
        plt.clf()
        
    del modelPlot
        
    return red_X_squared


############################ custom cmap for many distinct colors #################################

def generate_colormap(number_of_distinct_colors: int = 80):
    if number_of_distinct_colors == 0:
        number_of_distinct_colors = 80

    number_of_shades = 7
    number_of_distinct_colors_with_multiply_of_shades = int(math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades)

    # Create an array with uniformly drawn floats taken from <0, 1) partition
    linearly_distributed_nums = np.arange(number_of_distinct_colors_with_multiply_of_shades) / number_of_distinct_colors_with_multiply_of_shades

    # We are going to reorganise monotonically growing numbers in such way that there will be single array with saw-like pattern
    #     but each saw tooth is slightly higher than the one before
    # First divide linearly_distributed_nums into number_of_shades sub-arrays containing linearly distributed numbers
    arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades, number_of_distinct_colors_with_multiply_of_shades // number_of_shades)

    # Transpose the above matrix (columns become rows) - as a result each row contains saw tooth with values slightly higher than row above
    arr_by_shade_columns = arr_by_shade_rows.T

    # Keep number of saw teeth for later
    number_of_partitions = arr_by_shade_columns.shape[0]

    # Flatten the above matrix - join each row into single array
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

    # HSV colour map is cyclic (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic), we'll use this property
    initial_cm = hsv(nums_distributed_like_rising_saw)

    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half

    # Modify lower half in such way that colours towards beginning of partition are darker
    # First colours are affected more, colours closer to the middle are affected less
    lower_half = lower_partitions_half * number_of_shades
    for i in range(3):
        initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8/lower_half)

    # Modify second half in such way that colours towards end of partition are less intense and brighter
    # Colours closer to the middle are affected less, colours closer to the end are affected more
    for i in range(3):
        for j in range(upper_partitions_half):
            modifier = np.ones(number_of_shades) - initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i]
            modifier = j * modifier / upper_partitions_half
            initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i] += modifier

    return ListedColormap(initial_cm)



############################ mcmc behavior with custom cmap #################################


def plot_mcmc_behaviour_alt(ax, samples_mcmc, param_mcmc, dist_mcmc=None, num_average=100):
    """
    plots the MCMC behaviour and looks for convergence of the chain
    :param samples_mcmc: parameters sampled 2d numpy array
    :param param_mcmc: list of parameters
    :param dist_mcmc: log likelihood of the chain
    :param num_average: number of samples to average (should coincide with the number of samples in the emcee process)
    :return:
    """
    num_samples = len(samples_mcmc[:, 0])
    num_average = int(num_average)
    n_points = int((num_samples - num_samples % num_average) / num_average)
    
    n_params = len(param_mcmc)
    cmap=generate_colormap(n_params)
    colors = [cmap(i) for i in np.linspace(0, 1, n_params)]
    
    for i, param_name in enumerate(param_mcmc):
        samples = samples_mcmc[:, i]
        samples_averaged = np.average(samples[:int(n_points * num_average)].reshape(n_points, num_average), axis=1)
        end_point = np.mean(samples_averaged)
        samples_renormed = (samples_averaged - end_point) / np.std(samples_averaged)
        ax.plot(samples_renormed, label=param_name,color=colors[i])

    if dist_mcmc is not None:
        dist_averaged = -np.max(dist_mcmc[:int(n_points * num_average)].reshape(n_points, num_average), axis=1)
        dist_normed = (dist_averaged - np.max(dist_averaged)) / (np.max(dist_averaged) - np.min(dist_averaged))
        ax.plot(dist_normed, label="logL", linewidth=4,color='k')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return ax


############################ chainPlots #################################

def make_chainPlots(chain_list, chainPlot_path, num, object_ID):
    
    
    ####PSO chain plots
    chain_pso = chain_list[0]
    chain, param_list = chain_pso[1:]
    X2_list, pos_list, vel_list = chain
    f, axes = plt.subplots(2, 2, figsize=(22, 22))
    f.subplots_adjust(left=0.05, bottom=None, right=0.85, top=None, wspace=0.4, hspace=0.3)
    ax = axes[0,0]
    ax.plot(np.log10(-np.array(X2_list)))
    ax.set_title('-logL')

    ax = axes[0,1]
    pos = np.array(pos_list)
    vel = np.array(vel_list)
    n_iter = len(pos)
    
#     cmap = plt.get_cmap('rainbow')
    
    n_params = len(param_list)
    cmap=generate_colormap(n_params)
    colors = [cmap(i) for i in np.linspace(0, 1, n_params)]
    
    for i in range(0, len(pos[0])):
        ax.plot((pos[:, i]-pos[n_iter-1, i]) / (pos[n_iter-1, i] + 1), label=param_list[i],color=colors[i])
    ax.set_title('particle position')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax = axes[1,0]
    for i in range(0,len(vel[0])):
        ax.plot(vel[:, i] / (pos[n_iter-1, i] + 1), label=param_list[i],color=colors[i])
    ax.set_title('param velocity')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ### MCMC chain plot
    if len(chain_list) > 1:
        chain_mcmc = chain_list[1]
        samples, param, dist = chain_mcmc[1:]
    #     print(param)
        ax = axes[1,1]
        ax.set_title('mcmc behavior')
        plot_mcmc_behaviour_alt(ax, samples, param, dist, num_average=100)
    else: axes[1,1].set_axis_off()
        
    f.savefig(chainPlot_path + '/Image_{}-{}.png'.format(num, object_ID),dpi = 200)
    f.clear()
    plt.close(f)
    
    
############################ cornerPlots #################################

def make_cornerPlots(chain_list,cornerPlot_path,num,object_ID,step=1):
    sampler_type, samples_mcmc, param_mcmc, dist_mcmc  = chain_list[1]

    print("number of non-linear parameters in the MCMC process: ", len(param_mcmc))
    print("parameters in order: ", param_mcmc)
    print("number of evaluations in the MCMC process: ", np.shape(samples_mcmc[::step,:])[0])
    import corner
    if not samples_mcmc == []:
        n, num_param = np.shape(samples_mcmc[::step,::step])
        plot = corner.corner(samples_mcmc[::step,:], labels=param_mcmc[:], show_titles=True, title_fmt='.4f')
        plot.savefig(cornerPlot_path + '/Image_{}-{}.png'.format(num,object_ID),dpi=200)
        plot.clear()
        plt.close(plot)
########################## Save Chain List ###############################  
def save_chain_list(chain_list,chainList_path,num,object_ID):
    # Store data (serialize)
    with open(chainList_path + '/Image_{}-{}.pickle'.format(num, object_ID), 'wb') as handle:
        pickle.dump(chain_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        
        