# Data from Koul et al(2023)
Data was originally provided in the form of CSV files, which we converted into numpy(.npy) files and that is used for later analysis. We will also show the step in which we are converting the CSV files into numpy files, as at this stage, we had to use interpolation methods to supply missing values. Firstly, we defined three functions for interpolation. 
```python
def MovingAVG(array, winsize):                                                                                # For computing moving average
  avg = []
  for i in range(len(array) - winsize + 1):
    avg.append(np.sum(array[i:i+winsize]) / winsize)
    
  avg = np.array(avg)
  
  return avg
  
def Interpolate_time_series(data):                                                                          # For interpolating the data using PCHIP                  
    x = np.arange(len(data))
    valid_indices = np.where(np.isfinite(data))
    filled_data = data.copy()
    filled_data[np.isnan(data)] = pchip_interpolate(x[valid_indices], data[valid_indices], x[np.isnan(data)])
    return filled_data
   
def process_n_return_distances(path, x_column, y_column, winsize, distance):                               # This function does interpolation and moving average, returns either Euclidian or 
                                                                                                              angular distance
  data = pd.read_csv(path)
  
  x = np.array(data[x_column])
  if np.sum(np.isnan(x))>0:
    x = Interpolate_time_series(x)
    
  x = MovingAVG(x , winsize)
  y = np.array(data[y_column])
  if np.sum(np.isnan(y))>0:
    y = Interpolate_time_series(y)
    
  y = MovingAVG(y , winsize)
  
  print('nans in original x:',np.sum(np.isnan(x)))
  print('nans in original y:',np.sum(np.isnan(y)))
  
  if distance == 'Euclidian':
    dist= np.sqrt( (x*x) + (y*y))
    
  elif distance == 'angle': 
    dist= np.arctan(y/x)
    
  return dist

```
Now we will proceed with converting the csv files into numpy files



```python
column = 'RAnkle' 
y_column = column+'_y'
x_column = column+'_x'
path='bodymovements'
  
dyads = range(1,24)
conds = ['FaNoOcc','FaOcc','NeNoOcc','NeOcc']
trials = range(1,4)
sbjs = range(2)
  
DICT = {}
for cnd in conds:
  for trl in range(1,4):
    for dyad in range(1,24):
      dyd_files = []
      for sbj in range(2):
        path_sub = path +'/'+'results_video_'+str(sbj)+'_'+cnd+'_'+str(trl)+'_pose_body_unprocessed-'+str(dyad)+'.csv'
        dyd_files.append(path_sub)
          
      DICT['('+cnd+','+str(trl)+','+str(dyad)+','+column+')'] = dyd_files
          
        
comb_files = list(DICT.keys())
  

  
  
  
for KEY in tqdm(comb_files):
    
  data = []
  for File in DICT[KEY]:
    data.append(process_n_return_distances(File, x_column, y_column, 30, 'Euclidian'))
        
  data = np.array(data)
  data = data.T
  data = data[::6,:]
  np.save('/user/swarag/Koul_et_al/signals/('+cnd+','+str(trl)+','+str(dyad)+')~.npy',data)  # Save the data to a numpy file
  
```

Remaining steps are same, except that the vision condition(visual contact vs no visual contact) and proximity(near vs far) will be derived from variable "cnd". 
