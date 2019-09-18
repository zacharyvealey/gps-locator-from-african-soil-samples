# GPS Locator from African Soil Samples

Between 2009 and 2013, the Africa Soil Information Service (AfSIS) collected a large number of soil samples from 19 countries in Sub-Saharan Africa. The samples were georeferenced and subsequently analyzed for their nutrient content and chemical composition using a variety of methods that included spectroscopic and traditional wet chemistry techniques. A collation of this data is hosted at `arn:aws:s3:::afsis` where it can be downloaded for use with the Python code made available in this repository. The files presented here use the AfSIS dataset to pinpoint the GPS location of where the soil sample was taken based on the results from the associated “dry” and “wet” chemical analyses.

More information on the standard operating procedures of the soil sample collection and instrumental techniques can be found at https://github.com/qedsoftware/afsis-soil-chem-tutorial, which is associated with the hosted AWS dataset.

## Repository Information

Here are brief descriptions of the files made available in this repository:
* `afsis_soil_location_xray.py` contains the Python and scikit-learn code to predict the multi-target regression of latitude and longitude of soil sample location based on the results from X-ray fluorescence studies (which provide spectroscopically determined concentrations of elements).
* `afsis_soil_location_wet.py` contains the Python and scikit-learn code to predict the soil sample’s GPS coordinates as well, but instead uses the results from a much wider array of wet chemistry techniques that include extraction with Mehlich 3, pH determination, electrical conductivity analyses, exchangeable acidity measurements, soil moisture studies, etc.
* `afsis_soil_location_wet_and_dry.py` performs an analysis combing the information from the X-ray fluorescence results and wet chemistry data.
* `afsis_helper_functions.py` contains Python classes and functions that are used to assist the machine learning analyses in the above files.

## Summary of Results

Implementation of several machine learning algorithms revealed that k-nearest neighbors and random forest regression techniques performed the best for the wet chemistry and X-ray fluorescence data, respectively. The success of the different algorithms was measured by an R^2 metric with multi-target labelling of the latitude and longitude. (The wet chemistry data was also augmented by including the ratio between all element concentrations determined from Mehlich 3 extraction and then reduced in dimension through principal component analysis).

Evaluation of the R^2 and RMSE metrics on a test set of the data (after attendant hyperparameter tuning via random search) showed that use of the X-ray fluorescence results (R^2 = 0.8394, RMSE = 5.970deg) outperformed the use of the wet chemistry data (R^2 = 0.7590, RMSE = 7.558deg) signaling the advantage held by spectroscopic methods. Indeed, combination of both X-ray and wet chemistry data yielded only minor improvement over use of the X-ray data alone (R^2 = 0.8434, RMSE = 5.895deg).

Finally, examination of the feature importances determined from the random forest regression identified that the strongest predictors of the latitude and longitude of soil sample origin were the concentration of elements Y and K, respectively. 
