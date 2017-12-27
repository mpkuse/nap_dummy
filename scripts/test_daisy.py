import os
os.environ['LD_LIBRARY_PATH'] += './DaisyMeld' #might need to use path of nap package to set this correctly
from DaisyMeld.daisymeld import DaisyMeld

import numpy as np
dai = DaisyMeld()
im = np.random.random( (60,40) ) #float64
output = dai.hook( im.flatten(), im.shape )
output = np.array( output ).reshape( im.shape[0], im.shape[1], -1 )
