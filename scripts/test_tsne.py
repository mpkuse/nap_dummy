""" Tries with TSNE on higher dimensional data """

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import cv2
import rospkg

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import mpld3
import json
import code
import pickle

PKG_PATH = rospkg.RosPack().get_path('nap')
# PKG_PATH = '/home/mpkuse/catkin_ws/src/nap/'

def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()


    x, y = np.atleast_1d(x, y)
    artists = []
    i = 0
    for x0, y0 in zip(x, y):
        im = OffsetImage( cv2.cvtColor( image[i,:,:,:], cv2.COLOR_RGB2BGR ), zoom=zoom)
        i = i+1
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def load_json( file_name ):
    print 'Loading ', file_name
    with open( file_name ) as f:
        my_dict =  json.load(f)
        return  {int(k):float(v) for k,v in my_dict.items() }

#### LOAD Data ####
print 'Load Data'
folder = '/DUMP/tpt_night_loop/'
S_char = np.load( PKG_PATH+folder+'S_char.npy' )        #N x 128
S_word = np.load( PKG_PATH+folder+'S_word.npy' )        #N x 8192
S_thumbs = np.load( PKG_PATH+folder+'S_thumbnail.npy' ) #N x 96 x 128 x 3


#### Load Graph Data ####
print 'Load Groupping Graph Data'
internal_e = load_json( PKG_PATH+folder+'/internal_e.json' )
n_components = load_json( PKG_PATH+folder+'/n_components.json' )
#with open( PKG_PATH+folder+'all_nodes.pickle', 'r' ) as f:
#    all_nodes = pickle.load( f )

#### subset of key frames ####
print 'Subset of S_word'
key_frames =  sorted(internal_e)
key_frames.append( S_word.shape[0] )
# S_subset_indx = np.array( [k for k in key_frames[0:-1] ] )
S_subset_indx =  np.array([str(k) for k in range(0,S_word.shape[0],5) ])

# S_word_subset = np.array( [S_word[k,:] for k in key_frames[0:-1] ] )
S_word_subset = np.array( [S_word[k,:] for k in range(0,S_word.shape[0],5) ] )

# S_thumbs_subset = np.array([S_thumbs[k,:,:,:] for k in key_frames[0:-1] ])
S_thumbs_subset = np.array([S_thumbs[k,:,:,:] for k in range(0,S_word.shape[0],5) ])



## Dimensionality Reduction
# As suggested by Van der maaten's FAQ to about 50-100
# He suggested PCA for dense data and trucated SVD for sparse data
print 'Dim Red as suggested by Author of t-SNE'
pca = PCA( n_components=50 )
S_word_pca = pca.fit_transform( S_word_subset )
# S_word_pca = pca.fit_transform( S_word )




#### TSNE ####
print 't-SNE'
model = TSNE( n_components=2, random_state=3, perplexity=5, early_exaggeration=2, metric='cosine', verbose=1 )
# out = model.fit_transform( S_word )
out = model.fit_transform( S_word_pca )


#### Visualize #####
print 'plot'
fig, ax = plt.subplots()
# # imscatter(out[:,0], out[:,1], S_thumbs, zoom=0.5, ax=ax)
imscatter(out[:,0], out[:,1], S_thumbs_subset , zoom=0.5, ax=ax)
ax.plot(out[:,0], out[:,1], 'r.')
for label, x,y in zip( S_subset_indx, out[:,0], out[:,1] ):
    ax.annotate( label , xy=(x,y), xytext=(x,y), size=10 )
# plt.show()
mpld3.show()
