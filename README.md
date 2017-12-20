# ML_Final
ML final project, clickbait classifier
Data should be downloaded from the following website, unzipped and placed in the repository
http://www.clickbait-challenge.org/

Read the PDF document for our thought processes and journey through this challenge. 
Links to all data and references can be found on the last page of the PDF.
Feature extraction and fitting files are labelled as such. 


Feature extraction:
1. feature extraction for our thought up features: finalprojectextraction.py
2. full feature extraction, with mapping every character in title to an integer: finalprojectextraction_full.py
3. feature extraction, mapping every unique word in titles to an integer value: extraction_word_dict.py


Fitting methods:
1. fitting our features: finalprojectfitting_MnC.py
2. fitting the full data set, with each character of each title mapped to an integer: finalprojectfitting_full.py
3. attempts at fitting convolutional neural network, with the same features as #2: finalprojectfitting_conv.py
4. 2 layer 1D convolutional network for feature set #3: finalprojectfitting_word_dict_conv.py