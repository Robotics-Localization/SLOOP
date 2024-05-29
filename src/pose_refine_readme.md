1. Place `pose_estimation.py` and `plot_icp_result.py` in the same directory as the SLOOP.py program

2. Add a flag at the top of the main program

   ```python
   save_icp_result = 1
   ```
![image-20240527171624347](https://typora-img-bed--harold.oss-cn-beijing.aliyuncs.com/img/image-20240527171624347.png)

3. Add the following outside the loop where GM_Spherical_plane is located in the main program

   ```python
   pair_data_list = []
   if save_icp_result:
       icp_data_path = os.path.join(os.path.dirname(__file__), '..', f'data/icp_data')
       if not os.path.exists(icp_data_path):
           os.makedirs(icp_data_path)
   ```
![image-20240527171747886](https://typora-img-bed--harold.oss-cn-beijing.aliyuncs.com/img/image-20240527171747886.png)

4. Add the following after GM_Spherical_plane

   ```python
       import pose_estimation
       tf_init = pose_estimation.transform_two_coordinate_system(z_axis_src, y_axis_src, x_axis_src, origin_src, origin_2_src, z_axis_query, y_axis_query, x_axis_query, origin_query, origin_2_query)
       tf_icp = pose_estimation.vertex_icp_2d(vertexs_src, vertexs_query, tf_init)
       if save_icp_result:
           pair_data = {
               "src_vertex": vertexs_src,
               "que_vertex": vertexs_query,
               "src_frame": src_frame,
               "que_frame": query_frame,
               "init_transform": tf_init,
               "icp_transform": tf_icp,
           }
           pair_data_list.append(pair_data) 
   if save_icp_result:
       with open(os.path.join(icp_data_path, f'{seq:02d}.pkl'), 'wb') as file:
           pickle.dump(pair_data_list, file)
   ```
![image-20240527171949306](https://typora-img-bed--harold.oss-cn-beijing.aliyuncs.com/img/image-20240527171949306.png)

## Plotting the curve

After running the main program on all sequences, the pkl files containing poses results will be saved in the data/icp_data folder located one level above the main program directory. The data in the compressed file is the previously run results. Copy the folder from the compressed file to the directory one level above the main program directory, and you can directly run plot_icp_result.py to plot the graph.
