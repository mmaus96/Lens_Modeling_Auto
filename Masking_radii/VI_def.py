import pandas as pd 
import numpy as np 
import glob 
import matplotlib.pyplot as plt 
from astropy.io import fits
import math
import urllib.request
import os
from matplotlib.patches import Circle
from scipy.spatial import distance


def read_csv_files_1by1VI(path_csv_class,path_csv_prev,colnames):
	csv_files = glob.glob(path_csv_class+'*csv')
	csv_file_pred = glob.glob(path_csv_prev+'*csv')[0]
	df=pd.read_csv(csv_file_pred)
	dfp1 = pd.DataFrame()
	dfp2 = pd.DataFrame()
	i=0
	for i in range(len(csv_files)):
		rdf = pd.read_csv(csv_files[i])
		rdf = rdf.sort_values(by='file_name')
		rdf = rdf.reset_index()
		df['user'+str(i+1)+'_'+colnames[0]] = rdf['classification']
		df['user'+str(i+1)+'_'+colnames[1]] = rdf['subclassification']
		dfp1['user'+str(i+1)+'_'+colnames[0]] = rdf['classification']
		dfp2['user'+str(i+1)+'_'+colnames[1]] = rdf['subclassification']
		df = df.replace({'SA':'F'})
		dfp1 = dfp1.replace({'SA':'F'})
	return csv_files,df,dfp1,dfp2

def plot_freq(df,path,name):
	ind = np.arange(len(csv_files))
	names = ['user'+str(i+1) for i in range(len(csv_files))]
	barWidth = 0.75
	plt.figure(figsize=(8,8))
	freqL = []
	for i in range(len(csv_files)):
		freqL.append(len(df[df['user'+str(i+1)+'_'+colnames[0]]=='L']))
	plt.bar(ind, freqL, color='#4f5bd5', edgecolor='white', width=barWidth,label='L')
	freqML = []
	for i in range(len(csv_files)):
		freqML.append(len(df[df['user'+str(i+1)+'_'+colnames[0]]=='ML']))
	plt.bar(ind, freqML, bottom=freqL, color='#962fbf', edgecolor='white', width=barWidth,label='ML')
	barsF = np.add(freqL, freqML).tolist()
	freqF = []
	for i in range(len(csv_files)):
		freqF.append(len(df[df['user'+str(i+1)+'_'+colnames[0]]=='F']))
	plt.bar(ind, freqF, bottom=barsF, color='#d62976', edgecolor='white', width=barWidth,label='F')
	barsNL = np.add(barsF, freqF).tolist()
	freqNL = []
	for i in range(len(csv_files)):
		freqNL.append(len(df[df['user'+str(i+1)+'_'+colnames[0]]=='NL']))
	#plt.bar(ind, freqNL, bottom=barsNL, color='#fa7e1e', edgecolor='white', width=barWidth,label='NL')
	plt.xticks(ind, names, fontweight='bold')
	plt.legend(numpoints=1)
	labelL = [str(freqL[i]) for i in range(len(freqL))]
	for i in range(len(ind)):
		plt.text(x = ind[i] - 0.05, y = freqL[i]/2, s = labelL[i], size = 8)
	labelML = [str(freqML[i]) for i in range(len(freqML))]
	for i in range(len(ind)):
		plt.text(x = ind[i] - 0.05, y = (freqML[i])/2+freqL[i], s = labelML[i], size = 10)
	labelF = [str(freqF[i]) for i in range(len(freqF))]
	for i in range(len(ind)):
		plt.text(x = ind[i] - 0.05, y = (freqF[i])/2+freqL[i]+freqML[i], s = labelF[i], size = 10)
	plt.title('Main categories classified by user')
	plt.savefig(path+name+'.png')
	plt.close()
	return freqL,freqML,freqF,freqNL,ind

def plot_freq_sub(df,path,name,coln,n_users):
	ind = np.arange(n_users)
	names = ['user'+str(i+1) for i in range(n_users)]
	barWidth = 0.75
	plt.figure(figsize=(8,8))
	freqR = []
	for i in range(n_users):
		freqR.append(len(df[df['user'+str(i+1)+'_'+coln]=='Ring']))
	plt.bar(ind, freqR, color='#4f5bd5', edgecolor='white', width=barWidth,label='Ring')
	freqM = []
	for i in range(n_users):
		freqM.append(len(df[df['user'+str(i+1)+'_'+coln]=='Merger']))
	plt.bar(ind, freqM, bottom=freqR, color='#962fbf', edgecolor='white', width=barWidth,label='Merger')
	barsS = np.add(freqR, freqM).tolist()
	freqS = []
	for i in range(n_users):
		freqS.append(len(df[df['user'+str(i+1)+'_'+coln]=='Spiral']))
	plt.bar(ind, freqS, bottom=barsS, color='#d62976', edgecolor='white', width=barWidth,label='Spiral')
	plt.xticks(ind, names, fontweight='bold')
	plt.legend(numpoints=1)
	labelR = [str(freqR[i]) for i in range(len(freqR))]
	for i in range(len(ind)):
		plt.text(x = ind[i] - 0.05, y = freqR[i]/2, s = labelR[i], size = 8)
	labelM = [str(freqM[i]) for i in range(len(freqM))]
	for i in range(len(ind)):
		plt.text(x = ind[i] - 0.05, y = (freqM[i])/2+freqR[i], s = labelM[i], size = 10)
	labelS = [str(freqS[i]) for i in range(len(freqS))]
	for i in range(len(ind)):
		plt.text(x = ind[i] - 0.05, y = (freqS[i])/2+freqR[i]+freqM[i], s = labelS[i], size = 10)
	plt.title('Main categories classified by user')
	plt.savefig(path+name+'.png')
	plt.close()
	return freqR,freqM,freqS,ind



def df_inter_union(df,nameclas,cat,n_users):
	# inter
	dfinter = df
	for i in range(n_users-1):
		dfinter = dfinter[(dfinter['user'+str(i+1)+'_'+nameclas]==cat)&(dfinter['user'+str(i+2)+'_'+nameclas]==cat)]
	print('the intersection of all objects classified as '+cat+' is: ',len(dfinter))
	#union
	dfunion = pd.DataFrame()
	for i in range(n_users):
		dftemp=df[df['user'+str(i+1)+'_'+nameclas]==cat]
		dfunion=pd.concat([dfunion,dftemp],axis=1,sort=False)
	index=dfunion.index.values
	dfunionf = df.iloc[index]
	print('the union of all objects classified as '+cat+' is: ',len(dfunion))
	return dfinter, dfunionf

def percentages_percat(n_users,dfp_clas,df,path):
	perdf=dfp_clas.apply(pd.Series.value_counts,axis=1)/n_users
	print('ready perdf')
	#perdf.hist()
	#plt.savefig(path+'percentages_maincat.png')
	#plt.close()
	perdf = perdf.fillna(0)
	df['per_L'] = perdf['L']
	df['per_ML'] = perdf['ML']
	df['per_F'] = perdf['F']
	df['per_NL'] = perdf['NL']
	return df,perdf

def percentages_persubcat(n_users,dfp_clas,df):
	perdf=dfp_clas.apply(pd.Series.value_counts,axis=1)/n_users
	print('ready perdf')
	perdf = perdf.fillna(0)
	df['per_Ring'] = perdf['Ring']
	df['per_Merger'] = perdf['Merger']
	df['per_Spiral'] = perdf['Spiral']
	return df,perdf
	

def sum_mean(df,df_all):
	df=df.replace({'L':3,'ML':2,'F':1,'NL':0})
	m = df.mean(axis=1)
	s = df.sum(axis=1)
	df['mean'] = m
	df['sum'] = s
	df_all['mean'] = m
	df_all['sum'] = s
	return df,df_all

def comparison_among_users(n_users,df,nameclas,cat,freqcat,path):
	f1 = np.zeros(n_users*n_users).reshape(n_users,n_users)
	for j in range(n_users):
		for i in range(n_users):
			f1[j][i] = len(df[(df['user'+str(j+1)+'_'+nameclas]==cat)&(df['user'+str(i+1)+'_'+nameclas]==cat)])
	ind = np.arange(n_users) 
	plt.bar(ind,freqcat,width=0.1,label='own')
	[plt.bar(ind+(i+1)/10,f1[i],width=0.1,label='comp-U'+str(i+1)) for i in range(n_users)]
	label = [str(freqcat[i]) for i in range(n_users)]
	names = ['user'+str(i+1) for i in range(n_users)]
	plt.xticks(ind, names, fontweight='bold')
	plt.title('comparison among classifiers for the category '+str(cat))
	plt.legend(numpoints=1)
	plt.savefig(path+'comparison_'+str(cat)+'.png')
	plt.close()

def n_user_sameobj(n_users,dfper,path):
	dfper=dfper*6
	dffreq=dfper.apply(pd.Series.value_counts,axis=0).drop(index=0)
	ind = np.arange(n_users) 
	cl = ['L','ML','F','NL']
	for i in range(4):
		freqcat = dffreq[cl[i]].to_numpy()
		plt.bar(ind,freqcat)
		label = [str(freqcat[i]) for i in range(n_users)]
		names = [str(i+1)+' users' for i in range(n_users)]
		plt.xticks(ind, names, fontweight='bold')
		for j in range(n_users):
			plt.text(x = ind[j]-0.25 , y = freqcat[j]+2, s = label[j], size = 10)
		plt.title('# users that classified the same object as '+cl[i])
		#plt.show()
		plt.savefig(path+'n_users_'+cl[i]+'.png')
		plt.close()
	return dffreq

def sqrt_sc(inputArray, scale_min=None, scale_max=None):
    #this definition was taken from lenstronomy
    imageData = np.array(inputArray, copy=True)
    if scale_min is None:
        scale_min = imageData.min()
    if scale_max is None:
        scale_max = imageData.max()
    imageData = imageData.clip(min=scale_min, max=scale_max)
    imageData = imageData - scale_min
    indices = np.where(imageData < 0)
    imageData[indices] = 0.00001
    imageData = np.sqrt(imageData)
    imageData = imageData / np.sqrt(scale_max - scale_min)
    return imageData

def background_rms_image(cb,image):
    xg,yg = np.shape(image)
    cut0  = image[0:cb,0:cb]
    cut1  = image[xg-cb:xg,0:cb]
    cut2  = image[0:cb,yg-cb:yg]
    cut3  = image[xg-cb:xg,yg-cb:yg]
    l = [cut0,cut1,cut2,cut3]
    m=np.mean(np.mean(l,axis=1),axis=1)
    ml=min(m)
    mm=max(m)
    if mm > 5*ml:
    	s=np.sort(l,axis=0)
    	nl=s[:-1]
    	std = np.std(nl)
    else:
    	std = np.std([cut0,cut1,cut2,cut3])
    return std


def scale_val(image_array):
    if len(np.shape(image_array)) == 2:
        image_array = [image_array]
    vmin = np.min([background_rms_image(5,image_array[i]) for i in range(len(image_array))])
    xl,yl=np.shape(image_array[0])
    box_size = 14 #in pixel
    xmin = int((xl)/2-(box_size/2))
    xmax = int((xl)/2+(box_size/2))
    vmax = np.max([image_array[i][xmin:xmax,xmin:xmax] for i in range(len(image_array))])
    return vmin/2,vmax*2

def showplot_rgb(rimage,gimage,bimage):
    vmin,vmax=scale_val([rimage,gimage,bimage])
    img = np.zeros((rimage.shape[0], rimage.shape[1], 3), dtype=float)
    img[:,:,0] = sqrt_sc(rimage, scale_min=vmin, scale_max=vmax)
    img[:,:,1] = sqrt_sc(gimage, scale_min=vmin, scale_max=vmax)
    img[:,:,2] = sqrt_sc(bimage, scale_min=vmin, scale_max=vmax)
    return img

def showplot_rgb_HD(rimage,gimage,bimage):
    vmin,vmax=scale_val([rimage,gimage,bimage])
    img = np.zeros((rimage.shape[0], rimage.shape[1], 3), dtype=float)
    img[:,:,0] = sqrt_sc(rimage,scale_min=0, scale_max=np.max(gimage.flatten()))
    img[:,:,1] = sqrt_sc(gimage,scale_min=0, scale_max=np.max(gimage.flatten()))
    img[:,:,2] = sqrt_sc(bimage,scale_min=0, scale_max=np.max(gimage.flatten()))
    return img



def choose_subplot_dimensions(k):
    if k < 4:
        return k, 1
    elif k < 11:
        return math.ceil(k/2), 2
    else:
        # I've chosen to have a maximum of 3 columns
        return math.ceil(k/3), 3

def generate_subplots(k, ncol,row_wise=False):
    nrow=math.ceil(k/ncol)
    figure, axes = plt.subplots(nrow, ncol, figsize=(ncol*2,nrow*2),sharex=False,sharey=False)
    # Check if it's an array. If there's only one plot, it's just an Axes obj
    if not isinstance(axes, np.ndarray):
        return figure, [axes]
    else:
        # Choose the traversal you'd like: 'F' is col-wise, 'C' is row-wise
        axes = axes.flatten(order=('C' if row_wise else 'F'))
        # Delete any unused axes from the figure, so that they don't show
        # blank x- and y-axis lines
        for idx, ax in enumerate(axes[k:]):
            figure.delaxes(ax)
            # Turn ticks on for the last ax in each column, wherever it lands
            idx_to_turn_on_ticks = idx + k - ncol if row_wise else idx + k - 1
            for tk in axes[idx_to_turn_on_ticks].get_xticklabels():
                tk.set_visible(True)
        axes = axes[:k]
        return figure, axes

def mosaic(files,ncol,mosaic_name,mosaic_format='png',**kwargs):
	names = kwargs.get('names',[None])
	var = kwargs.get('var',[None])
	vname = kwargs.get('vname',[None])
	nname = kwargs.get('nname',[None])
	figure, axes = generate_subplots(len(files), ncol,row_wise=True)
	ind = np.arange(len(files))
	for files, ax, ind in zip(files, axes,ind):
		im= fits.open(files)
		if len(im) == 1:
			im0= im[0].data
			vmin,vmax=scale_val([im0])
			image=sqrt_sc(im0, scale_min=vmin, scale_max=vmax)
			ax.imshow(image,cmap='Greys_r',origin='lower')
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			ax.text(len(image)/2, 50-3.5, str(vname)+' = '+str(np.round(var,3)), horizontalalignment='center',fontsize=10,color='w')
			if names != None:
				ax.text(len(image)/2, 4.5, names, horizontalalignment='center',fontsize=10,color='w')
			if var != None:
				ax.text(len(image)/2, 50-3.5, str(vname)+' = '+str(np.round(var,3)), horizontalalignment='center',fontsize=10,color='w')
		if len(im) == 3:
			G,R,I = im[0].data,im[1].data,im[2].data
			color_img = showplot_rgb(I,R,G)
			ax.imshow(color_img,origin='lower')
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			if names[0] != None:
				ax.text(len(color_img)/2, len(color_img)-5, str(nname)+names[ind], horizontalalignment='center',fontsize=10,color='w')
			if var[0] != None:
				ax.text(len(color_img)/2, 5, str(vname)+str(np.round(var[ind],3)), horizontalalignment='center',fontsize=10,color='w')
	plt.tight_layout()
	plt.savefig(mosaic_name+'.'+mosaic_format,facecolor='black')
	plt.close()	



def get_legacy_survey(df,path,start):
	sample=df
	print(sample)
	savedir = path
	for i in range(start,len(sample)):
		ra = sample.iloc[i]['ra']
		dec = sample.iloc[i]['dec']
		#url = 'http://legacysurvey.org/viewer/cutout.jpg?ra=' + str(ra) + '&dec=' + str(dec) + '&layer=dr8&pixscale=0.6'
		url = 'http://legacysurvey.org/viewer/cutout.jpg?ra=' + str(ra) + '&dec=' + str(dec) + '&layer=dr8&size=500'
		savename = 'N' + str(i)+ '_' + str(ra) + '_' + str(dec) + 'dr8.jpg'
		urllib.request.urlretrieve(url, savedir + savename)
		url = 'http://legacysurvey.org/viewer/cutout.jpg?ra=' + str(ra) + '&dec=' + str(dec) + '&layer=des&pixscale=0.06'
		savename = 'N' + str(i)+ '_' + str(ra) + '_' + str(dec) + 'dr8-resid.jpg'
		urllib.request.urlretrieve(url, savedir + savename)

def duplicates(dfori,deg,mindup,path=None,savefiles=None):
	df = dfori
	dupdf = pd.DataFrame(columns=df.columns)
	for i in range(len(df)):
		df['rad'] =abs(df['ra'].iloc[i]-df['ra'])
		df['decd'] = abs(df['dec'].iloc[i]-df['dec'])
		dup=df[(df['rad']<deg) & (df['decd']<deg) | (df['rad']==0) | (df['decd']==0) ]
		if len(dup)>mindup:
			dupdf = pd.concat([dupdf,dup],join='outer')
			if savefiles == True:
				dup.to_csv(path+str(dup.iloc[0].id)+'_dupfile.csv')
		dupdf=dupdf.drop_duplicates(subset=['ra'])
	return dupdf 


def mosaic_mask(files,ncol,mosaic_name,dst,d_extra,cen,mosaic_format='png',**kwargs):
	names = kwargs.get('names',[None])
	var = kwargs.get('var',[None])
	vname = kwargs.get('vname',[None])
	nname = kwargs.get('nname',[None])
	figure, axes = generate_subplots(len(files), ncol,row_wise=True)
	ind = np.arange(len(files))
	for files, ax, ind in zip(files, axes,ind):
		im= fits.open(files)
		circle = Circle((cen,cen), dst[ind]+d_extra,facecolor='none',edgecolor='white', linewidth=3, alpha=0.5)
		ax.add_patch(circle)
		if len(im) == 1:
			im0= im[0].data
			vmin,vmax=scale_val([im0])
			image=sqrt_sc(im0, scale_min=vmin, scale_max=vmax)
			ax.imshow(image,cmap='Greys_r',origin='lower')
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			ax.text(len(image)/2, 50-3.5, str(vname)+' = '+str(np.round(var,3)), horizontalalignment='center',fontsize=10,color='w')
			if names != None:
				ax.text(len(image)/2, 4.5, names, horizontalalignment='center',fontsize=10,color='w')
			if var != None:
				ax.text(len(image)/2, 50-3.5, str(vname)+' = '+str(np.round(var,3)), horizontalalignment='center',fontsize=10,color='w')
		if len(im) == 3:
			G,R,I = im[0].data,im[1].data,im[2].data
			color_img = showplot_rgb(I,R,G)
			ax.imshow(color_img,origin='lower')
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			if names[0] != None:
				ax.text(len(color_img)/2, len(color_img)-5, str(nname)+names[ind], horizontalalignment='center',fontsize=10,color='w')
			if var[0] != None:
				ax.text(len(color_img)/2, 5, str(vname)+str(np.round(var[ind],3)), horizontalalignment='center',fontsize=10,color='w')
	plt.tight_layout()
	plt.savefig(mosaic_name+'.'+mosaic_format,facecolor='black')
	plt.close()	
