# creating the main of the programm
#this is the heart of the programm here the programm gona start running
from typing import Dict
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import StandardScaler
from visualization import visualization
from dataHandler import aggregation ,ImportData, DataCleaning



def main():
    data = ImportData.to_dataframe()
    user_act = aggregation.user_activity(data)
    visualization.user_activity(user_act)
    
    
    
    
    
    
    
if __name__ == "__main__":
    main()