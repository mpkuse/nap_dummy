# How to Launch

## General Node Organization
**NUC**
- vins_estimator
- feature_tracker
- pose_graph_optimization 
- rviz
- nap_dummy 

**tx2**
- nap_robustdaisy
- geometry_node


## Launch
**on NUC**
- roscore
- export ROS_MASTER_URI=http://urop:11311/
- roslaunch nap point_gray.launch 

**on tx2**
- ROS_MASTER_URI=http://urop-laptop:11311/
- set DaisyMeld flags
- roslaunch nap tx2.launch
