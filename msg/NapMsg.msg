time c_timestamp
time prev_timestamp
float32 goodness
float32 color_r
float32 color_g
float32 color_b


# Operation Mode
# 10 : You need to compute pose/ matching urself 
# 20 : Expansion (not implemented as of 26th Oct, 2017) uv xyz u'v'
# 30 : Contains relative pose (computed with 3-way matching)
# 29 : Contains 3-way matching. This is for internal consumption for mpkuse
int32 op_mode 

# MODE29 ########################################
# Point matching related. Can ignore if not using 
int32 n_sparse_matches

# For 3way match. Only uses x,y. This represents point in an image 
geometry_msgs/Point32[] curr
geometry_msgs/Point32[] prev
geometry_msgs/Point32[] curr_m
time t_curr
time t_prev
time t_curr_m
#########################################



## Mode30 ######################################
geometry_msgs/Pose p_T_c # Relative pose of current in frame of previous. Refer to timestamps of current and prev
