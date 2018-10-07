# forecast-aftershock-location

## About
Aftershocks can be as dangerous as an earthquakes, sometimes even more devastating. These are produced by the stress that was caused by the earthquake. Now, by predicting the location of these aftershocks we can save a lot of damage to human life and resources. Currently  we have been using the coulomb's stress criterion to explain the spatial distributions of aftershocks, but as the advent of technology is improving, it is highly possible that these machine learning models can find a undiscovered pattern that can be helpful in prediciting the fair locations of aftershocks.

## How will it work?
The model will use deep learning techniques to predict the aftershock locations and its severity. I'll be taking the data provided by the SRCMOD http://equake-rc.info/SRCMOD/.
The data provided by the SCRMOD contains the information about the hypocenter, latitude, longitude, magnitude, strike, dip and most importantly the inversion parameters and many other relevant data that is sufficient to get an insight of the earthquake and the probable location of the aftershocks. So we'll be feeding these data to the neural network having several hidden layers (not decided yet), which will then try to extract the important features in order to predict the longitude and latitude of the aftershocks. we'll be using two activation functions tanh and ReLu (not experimented yet). The ouput layer will contain three neurons two of them will predict the locations and the last one will predict the severity.
<br>

## How will it help?
The above model according to me can be of great help to the society, as sometimes it is not the earthquake that do a lot damage but the incoming aftershocks. so if we can get an insight about the stress pattern that it left to the area, we can farely warn people about the chances of having an aftershock at that particular location.
