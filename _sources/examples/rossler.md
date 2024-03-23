# Rossler Attractor
Firstly, we will define functions for simulating Rossler attractor


```python
def rossler(t, X, a, b, c):
    """The Rössler equations."""
    x, y, z = X
    xp = -y - z
    yp = x + a*y
    zp = b + z*(x - c)
    return xp, yp, zp
    
def rossler_time_series(tmax,n, Xi, a, b, c):
    x, y, z = Xi
    X=[x]
    Y=[y]
    Z=[z]
    dt=0.0001
    for i in range(1, tmax):
      #print('Xi:', Xi)
      x, y, z = Xi
      xp, yp, zp=rossler(t, Xi, a, b, c)
      x_next=x+dt*xp
      y_next=y+dt*yp
      z_next=z+dt*zp
      X.append(x_next)
      Y.append(y_next)
      Z.append(z_next)
      Xi=(x+dt*xp,y+dt*yp,z+dt*zp)
      
    X=np.array(X)
    Y=np.array(Y)
    Z=np.array(Z)
    
    step=int(tmax/n)
    indices = np.arange(0,tmax, step)
    #print('IS NAN X:',np.sum(np.isnan(X[indices])))
    #print('IS NAN Y:',np.sum(np.isnan(Y[indices])))
    #print('IS NAN Z:',np.sum(np.isnan(Z[indices])))
    
    return X[indices], Y[indices], Z[indices]
```
Now, define the parameter values. We are mainly varying a, while keeping b and c the same for getting transition from periodic to chaotic attractor. 
```python
b = 0.2
c = 5.7
a_array = [0.1,0.15,0.2,0.25,0.3]
SNR = [0.125,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]
```

Now we will simulate the data for different nose levels, and will save it to a directory


```python
for snr in tqdm(SNR):                                                                             # Looping over SNR values
  for j in tqdm(range(len(a_array)):                                                              # Looping over values of parameter a
    a=a_array[j]
    random.seed(1)
    np.random.seed(seed=301)
    rp_sizes = random.sample(range(150, 251), 10)                                                 # selecting a length for time series
    for k in tqdm(range(5)):                                                                      # Repeats
       
          u0, v0, w0 = 0+(1*np.random.randn()), 0+(1*np.random.randn()), 10+(1*np.random.randn()) # Setting initial conditions
          
          
          for k2 in tqdm(range(len(rp_sizes))):                                                   # Looping over different RP sizes(time series length)
            tmax, n = int(1000000*(rp_sizes[k2]/250)), rp_sizes[k2]
            print('started model')
            Xi=(u0, v0, w0)
            t = np.linspace(0, tmax, n)
            x, y, z=rossler_time_series(tmax,n, Xi, a, b, c)
            
            x=add_noise(x, snr)                                                                  # Adding noise
            y=add_noise(y, snr)
            z=add_noise(z, snr)
            u[:,0]=x                                                                             # Defining the output matrix
            u[:,1]=y
            u[:,2]=z
            np.save('/user/swarag/Rossler/signals/('+str(snr)+','+str(a)+','+str(u0)+','+str(v0)+','+str(w0)+','+str(rp_sizes[k2])+')~.npy',u)  # Save the data to a numpy file
```
Since we have saved the time series data to a folder, we can repeat the same steps


```python
input_path = '/user/swarag/Rossler/signals'                                        # directory to which the signals are saved
RP_dir = '/user/swarag/Rossler/RP'                                                 # directory to which we want to save the RPs
RP_computer(input_path, RP_dir)                                                    # generating RPs and saving to the specified folder
``

Since we have generated the RPs at this step, we can extract the variables from it. We have selected same window size for the sliding window approach. 


```python
Dict_RPs=windowed_RP(68, 'RP')                                                      # Specifying window size and folder in which RPs are saved
First_middle_last_sliding_windows_all_vars(Dict_RPs,'Rossler_data.csv')            # Saving RQA variables to a csv file
```

Now we need to add additional columns to the data for later analysis


```python
data = pd.read_csv('Rossler_data.csv')
FILE = np.array(data['group'])                                                      # In the output data, the field named 'group' will have file name which contains details
SNR =[]
A =[]
U0 =[]
V0 =[]
W0 =[]
SIZE =[]
for FI in FILE:
  info = ast.literal_eval(FI)
  SNR.append(info[0])
  A.append(info[1])
  U0.append(info[2])
  V0.append(info[3])
  W0.append(info[4])
  SIZE.append(info[5])
  
data['snr'] = SNR
data['a'] = A
data['u0'] = U0
data['v0'] = V0
data['w0'] = W0
data['length'] = SIZE
A = np.array(A)

SYNCH = 1*(A>0.2)                                                                   # Defining synchrony condition, parameter value belonging to the chaotic region
data['synch'] = SYNCH
```
Now, we need to select an SNR value, scale the variables and run the classifier

```python
################################################################## Select the value of SNR ################################################################################################
data_ = data[data['snr']==1.0].reset_index(drop = True)
data_ = data_[data['window']== 'mode'].reset_index(drop = True)                   # mode of RQA variables from the sliding windows
################################################################## Scale the data #########################################################################################################\
features=['recc_rate',
 'percent_det',
 'avg_diag',
 'max_diag',
 'percent_lam',
 'avg_vert',
 'vert_ent',
 'diag_ent',
 'vert_max']
for feature in features:
  arr = np.array(data_[feature])
  data_[feature] = (arr - np.mean(arr))/(np.std(arr) + 10**(-9))
  
################################################################# Run the classification ###################################################################################################
nested_cv(data_, features, 'synch', 'Kuramotot(SNR=1.0)', repeats=100, inner_repeats=10, outer_splits=3, inner_splits=2)
#################################################################   DONE ! ################################################################################################################
```
