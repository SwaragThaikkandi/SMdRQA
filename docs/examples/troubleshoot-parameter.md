# Troubleshooting: Parameter Search for embedding dimension

If you were suspecious about the oversimplified look of the function "RP_computer", you are correct. This function also implements a parameter search method for finding the embedding dimensions under the hood. Even though some default values are there, they may not be applicable to all datasets. In such cases, we need to see, what can be set. We had to do this while analyzing data from Koul et al(2023), and it is for this, we have some functions in "RP_maker_diagnose.py". We will see how we can use these functions


```python
from RP_maker_diagnose import fnnhitszero_Plot
from RP_maker_diagnose import findm_Plot
from RP_maker_diagnose import RP_diagnose
from RP_maker_diagnose import get_minFNN_distribution_plot

input_path = '/user/swarag/Koul et al/data_npy'                                        # directory to which the signals are saved
diagnose_dir = '/user/swarag/Koul et al/diagnose'                                            # directory in which pickle files from this function are saved
RP_diagnose(input_path, diagnose_dir)
```
Now, we have saved the picke files to a directory, we can use those pickle files in the next function. Following function estimates the lower and upper bound (2.5th and 975th percentile) of the distribution of minimum false nearest beighbour(FNN) values for different embedding dimensions. It gives a plot and a CSV file. 
```python
get_minFNN_distribution_plot(path, 'Koul_et_al_RP_diagnose')
```

<div style="position: relative; width: 100%;">
  <img src="https://github.com/SwaragThaikkandi/Sliding_MdRQA/blob/main/Koul_et_al_RP_diagnose.png" style="width: 750px;">
  
</div>
<div style="position: relative; width: 100%;">
  <img src="https://github.com/SwaragThaikkandi/Sliding_MdRQA/blob/main/Koul_et_al_RP_diagnos_csv.png" style="width: 300px;">
  
</div>
We can see that the minimum value of FNN goes down as we increases the embedding diension. But, the value of delta, which is a parameter that determines whether a particular value of embedding dimnesion can be effectively considered as zero or not, should include most of these values. Generally m=1 is less useful, hence, we can consider the upper bound from m=2 onwards. Then we can set this limit for defining the r value at which FNN hits zero and then we can find out desired value of embedding dimension by setting a threshold value for r(at which FNN hits zero) vs embedding dimension plot.
