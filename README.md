# forecast-aftershock-location

## About
Aftershocks can be as dangerous as an earthquakes, sometimes even more devastating. These are produced by the stress that were caused by the earthquake. Now, by predicting the location of these aftershocks we can save a lot of damage to human life and resources, because most of the time it is these aftershocks that do a lot damage than main shocks. Currently we have been using the coulomb's stress criterion to explain the spatial distributions of aftershocks, but as the advent of technology is improving, it is highly possible that these machine learning models can find a undiscovered pattern that can be helpful in prediciting the fair locations of aftershocks.

## How will it work?
The idea is to observe a volume which extends 100 km horizontally and 50 km vertically from the main shock. We then break that volume into 5km x 5km x 5km small volumes and calculate the elastic stress change tensors at each of their centroid. Now using that information we need to predict whether there was an aftershock in that small volume or not. In order to know the ground truth we use International Seismological Center (ISC) event catalogue, in which for each main shock we looked up for its corresponding aftershocks from 1 sec to 1 year time and using that information we created our ground truth i.e. whether there was an aftershock in that small volume or not.
So this whole problem is now a binary classification problem, in which our neural network has to predict was there an aftershock in that small 5km x 5km x 5km region or not.
The model will use deep learning techniques to predict whether there can be aftershock at a particular locations or not. I'll be taking the data provided by the SRCMOD http://equake-rc.info/SRCMOD/searchmodels/allevents/. The format of the data will be FSP (finite-source rupture model). We'll not be working on how to process those SRCMOD files in order to create a csv. Rather we'll use already created CSVs.
Let me give you a brief description of the CSV files. Each csv file contains <b>16</b> columns, 
<ul>
  <li><i> <b>x</b>, <b>y</b> and <b>z</b> </i>are the coordinate of the centre of the grid i.e. the small volumes. </li> 
  
  <li> <b><i> stresses_full_xx, stresses_full_xy, stressess_full_yy, stressess_full_xz, stressess_full_yz, stressess_full_zz</i></b> these are the magnitudes of the six independent components of the co-seismically generated static elastic stress-change tensor calculated at the centroid of a grid cell. </li>
  
  <li> <b><i>stresses_full_cfs_1, stresses_full_cfs_2, stresses_full_cfs_3, stresses_full_cfs_4</i></b> these are coloumbic failure stress, you can notice that there are only two column that are distinct in magnitude other two are just having the opposite signs.</li>
  <li> <i><b> aftershocksyn </b></i> it is our dependent variable, which we have to predict for each grid</li>
</ul>
The data provided by the SCRMOD file contains the information about the hypocenter, latitude, longitude, magnitude, strike, dip, the inversion parameters and many other relevant data that is sufficient to get an insight of the earthquake. 
Now, as we've our CSV file ready, we're good to go...
We'll be feeding these data to the neural network having several hidden layers (it depends on you, try and experiment with it), which will then try to extract the important features in order to predict the whether there is a chance of having an aftershock or not. we'll be using two activation functions tanh and ReLu (again experiment with it). The ouput layer will be a sigmoid layer, which will give us a probability between 0 - 1, as to how confident it is.
<br>

## How will it help?
The above model according to me can be of great help to the society, as sometimes it is not the earthquake that do a lot damage but the incoming aftershocks. so if we can get an insight about the stress pattern that it left to the area, we can farely warn people about the chances of having an aftershock at that particular location.

## References
<i>link to the paper</i> https://www.nature.com/articles/s41586-018-0438-y
<br>
<i>link to the csv files</i> https://drive.google.com/drive/folders/1c5Rb_6EsuP2XedDjg37bFDyf8AadtGDa?usp=sharing
<br>
<i>link to relevant materials</i> http://www.bosai.go.jp/study/application/dc3d/DC3Dhtml_E.html

## Steps

follow the instruction given below to get started with the model<br>
```
git clone https://github.com/KishanChandravanshi/forecast-aftershock-location.git
cd forecast-aftershock-location
```

<br>

open your command prompt in this folder, and install all the dependencies provided in the <b> requirement.txt </b> file.

`pip install -r requirements.txt`
<br>
* Download the required csv files from the link provided and place them in <b> csvfiles </b> folder.
* Download the required .h5 files from the link and place them inside the <b> Data </b> directory.
* Now run <b> buldData.py </b> from the terminal or CMD.<br>
  <i> buildData.py will create a training_tmp.h5 and testing_tmp.h5 files, these will be the file that you can use for training the models. We're first creating the .h5 file instead of using the data directly for training to reduce the time and memory consumption by the data, because the data is quite big, it will take a substantiate amount of memory, while training, but by creating a .h5 files we are reducing the loading and reading process time by a large fraction.</i>
* After executing has finished, run <b>trainModel.py</b> file from cmd or terminal. Note you can either use the weights downloaded from the internet to train your model or you can use your created <b> training_tmp.h5 </b> for this purpose, just place them in the Data directory and you are good to go.

* To evaluate your model on the test set, execute <b>evaluate.py </b>, it will give you roc_auc_score
* <b> helper_function.py </b> as the name implies contain all the function that are being required in the other files.
