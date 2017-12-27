# Neural Appearence based Place model (NAP)
==============================================

## Basic Steps
This ros package implements the NAP. Basic steps involved   
1. Retrive image   
2. Normalize colors (R=R/(R+G+B), G=G/(R+G+B), B=B/(R+G+B))   
3. Compute netVLAD (8192-d)    
4. Dimensionality reduction (Dim Red by learning invariant mapping) (128-d)    
5. Store descriptor in array    
6. simScore = Dot( curDesc, all prev desc )    
7. visualize simScore with rviz    

## Neural Data
1. To compute netvlad discriptor (8192-dim) using PARAM\_MODEL    
2. To compute invariant mapping (dim-red) using PARAM\_MODEL\_DIM\_RED    


<code>
PARAM_MODEL = PKG_PATH+'/tf.logs/netvlad_angular_loss_w_mini_dev/model-4000'    
PARAM_MODEL_DIM_RED = PKG_PATH+'/tf.logs/siamese_dimred_fc/model-400'
</code>


## Files
### Main Scripts
1. nap\_debug\* --> These are the scripts which implement the said algorithms. Images read from published image. GT loaded.
2. im\_generator\_malaga.py --> Script to read from malaga-like seq and publish images
3. im\_generator\_kitti.py --> Read from kitti like datasets

### Helper Classes
1. CartWheelFlow.py --> Construction of netvlad neural net
2. DimRed.py --> related to invariant mapping (Dim-red)
3. Quaternion.py --> Quaternion helper class
4. VPTree.py --> Incremental insertion (non-balanced) and nn-search in a Vantage point tree

## Author
Manohar Kuse <mpkuse@connect.ust.hk>


