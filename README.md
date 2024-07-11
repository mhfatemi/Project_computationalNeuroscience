# Project-Computational-Neuroscience
This repository is about project of computational neuroscience 

## 1.Problem statement
  
Decoding arm motion using electrocorticographic (ECoG) signals in monkeys using CNN, LSTM, and PLS.

## 2.Related works
  explores existing research and solutions related to your project

## 3.Proposed Method
Feature extraction using Morlet wavelet transform and decoding with neural networks and PLS.

## 4.Implementation

### 4.1. Dataset
Under this subsection, you'll find information about the dataset used for your project. It includes details about the dataset source,task, channels, and number of trials. 
http://neurotycho.org/expdatalist/listview?task=36
ECoG signals and arm motion were recorded during a food tracking task in two Japanese macaques. Signals from Monkey A were recorded using 32 electrodes, and signals from Monkey K were recorded using 64 electrodes. The two monkeys together performed 35 trials.
### 4.2. Model
We used LSTM and CNN as deep learning models, and PLS as a classical regression model for decoding. Morlet wavelet transform was used for feature extraction.

### 4.3. Evaluate
To evaluate the efficiency of the decoding model, we calculated the correlation coefficient between the predicted and observed trajectories.
