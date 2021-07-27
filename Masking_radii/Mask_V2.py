import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from matplotlib.patches import Circle
from scipy.spatial import distance
from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder
from photutils import CircularAperture
import sys
# sys.path.insert(1,'/home/karina/Dropbox/postdoc_EPFL/ANN/CNNstudentproject/CNN_200k/classification_mosaic/known_candidates/catalogs/')
import VI_def as vid
import progressbar
from astropy.table import QTable

# path = '/home/karina/Dropbox/postdoc_EPFL/ANN/CNNstudentproject/CNN_200k/classification_mosaic/eso_prop/'
# path_fits = '/home/karina/Dropbox/postdoc_EPFL/ANN/CNNstudentproject/CNN_200k/candidates2/'

path = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/'
path_fits = '/Users/markmaus/Desktop/Physics_EPFL/Specialization_Project/lens_candidates/Sure_Lens/data/'


df=pd.read_csv(path+'lenses05.csv')
df = df[df['class_fVI_1']=='L']



df['dst_arcsec'] = 100
df['dst_pix'] = 100

bar = progressbar.ProgressBar(maxval=len(df), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()

for j in range(len(df)):
	bar.update(j)
	im= fits.open(path_fits+df['file_name'].iloc[j])
	mean, median, std = sigma_clipped_stats(im[0].data, sigma=3.0)
	daofind = DAOStarFinder(fwhm=5.0, threshold=5.*std)
	sources = daofind(im[0].data - median)
	if sources == None:
	    daofind = DAOStarFinder(fwhm=8.0, threshold=5.*std)
	    sources = daofind(im[0].data - median)
	if sources == None:
		x = [0,15]
		y = [0,15]
		sources = QTable([x,y],names={'xcentroid','ycentroid'})
	sources['difx'] = abs(sources['xcentroid']-26)
	sources['dify'] = abs(sources['ycentroid']-24)
	s = sources.to_pandas()
	print(s)
	s['md'] = s[['difx','dify']].mean(axis=1)
	s = s.sort_values(by=['md'])
	s = s.reset_index()	
	if (len(s)>1) and (s['md'][0]<3):
	    print('d1',s['xcentroid'].values[1],s['ycentroid'].values[1])
	    dst=distance.euclidean((25,25),(s['xcentroid'].values[1],s['ycentroid'].values[1]))
	    print(dst)
	    if (s['md'][1]>12):
	        print('d2')
	        dst=distance.euclidean((25,25),(s['xcentroid'].values[0],s['ycentroid'].values[0]))   
	else:
	    print('re-c')
	    daofind = DAOStarFinder(fwhm=8.0, threshold=5.*std)
	    sources = daofind(im[0].data - median)
	    sources['difx'] = abs(sources['xcentroid']-26)
	    sources['dify'] = abs(sources['ycentroid']-24)
	    s = sources.to_pandas()
	    s['md'] = s[['difx','dify']].mean(axis=1)
	    s = s.sort_values(by=['md'])
	    s = s.reset_index()
	    print(s)
	    print(len(s))
	    if (len(s)==1):
	        print('d3')
	        dst=distance.euclidean((25,25),(s['xcentroid'].values[0],s['ycentroid'].values[0]))
	    if (len(s)>1) :
	        if (s['md'][1]>12):
	            print('d4')
	            dst=distance.euclidean((25,25),(s['xcentroid'].values[0],s['ycentroid'].values[0]))
	        else:
	            print('d5')
	            dst=distance.euclidean((25,25),(s['xcentroid'].values[1],s['ycentroid'].values[1]))
	df['dst_pix'].iloc[j] = dst
	df['dst_arcsec'].iloc[j] = dst*0.27


bar.finish()


df.to_csv(path+ 'Sure_Lens/' + 'mask_v2.csv')

#*****************
#	Mosaic
#*****************

df=pd.read_csv(path+'mask_v2.csv')

vid.mosaic_mask(path_fits+df['file_name'],12,path+'mask_v2',df['dst_pix'],8,25,var=df['dst_arcsec'],vname='TE = ')
