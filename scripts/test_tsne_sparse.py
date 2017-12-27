""" T-SNE on sparse matrix data of dim Nx900000. This matrix is
    the BOW representation of every image. This data is stored in json
        [
            {"vocabulary_size": 3222},
            {"0.id": [2, 3, 6, 7]},
            {"0.wt": [.22, .3, .16, .7]},
            {"1.id": [3, 10]},
            {"1.wt": [.23, .10]},
            {"2.id": [2, 21]},
            {"2.wt": [.22, .21]}
        ]

    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
    Created : 31st May, 2017
"""

import numpy as np
import cv2
import rospkg

from sklearn.manifold import TSNE
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import json
import code
import pickle

PKG_PATH = rospkg.RosPack().get_path('nap')

def load_bow_as_sparse( file_name ):
    e = json.loads( open(file_name).read() )
    vocab_size = e['vocabulary_size']
    total_images = e['total_images']
    print 'Open file : ', file_name
    print 'vocab_size :', vocab_size
    print 'total_images : ', total_images


    total_images_sze = int(np.ceil( total_images / 10. ))
    A = lil_matrix( (  total_images_sze, vocab_size) )
    thumbs = []
    thumbs_indx = []
    c=0
    for i in range(0,total_images,10):
        im = np.array( e[str(i)+'.im'] ).astype( 'uint8' )
        thumbs.append( im )
        thumbs_indx.append( i )

        ind_list = e[str(i)+'.id']
        ind_wt = e[str(i)+'.wt']
        for l in range(len(ind_list)):
            A[c, ind_list[l]] = ind_wt[l]
        c=c+1
    return A, np.array(thumbs), thumbs_indx

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


#### Load Data ###
print 'Loading Sparse Matrix'
folder = '/DUMP/tpt_night_loop/'
S_bowvec, S_thumbs, S_thumbs_indx = load_bow_as_sparse( PKG_PATH+folder+'/dbow_per_image_SKIP0.json' )


## Dimensionality Reduction
# As suggested by Van der maaten's FAQ to about 50-100
# He suggested PCA for dense data and trucated SVD for sparse data
print 'TruncatedSVD for dim-red'
svd = TruncatedSVD(n_components=200, n_iter=7, random_state=42, tol=0.0 )
S_bowvec_svd = svd.fit_transform( S_bowvec )



#### TSNE ####
print 't-SNE'
model = TSNE( n_components=2, random_state=3, perplexity=20, early_exaggeration=1, metric='cosine', verbose=1 )
# out = model.fit_transform( S_bowvec )
out = model.fit_transform( S_bowvec_svd )


#### Visualize #####
print 'plot'
fig, ax = plt.subplots()
imscatter(out[:,0], out[:,1], S_thumbs, zoom=0.5, ax=ax)
ax.plot(out[:,0], out[:,1], 'r.')
for label, x,y in zip( S_thumbs_indx, out[:,0], out[:,1] ):
    ax.annotate( label , xy=(x,y), xytext=(x-3,y-3), size=10 )
plt.show()
