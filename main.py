# creating the main of the programm
#this is the heart of the programm here the programm gona start running
from visualization import visualization
from dataHandler import aggregation ,ImportData, DataCleaning, PreProcessData
from MoovieRecommendations import TrainModels


def main():

    #import the data and make it dataframe
    #ImportData.import_data()
    data_dict = ImportData.to_dataframe()
    
    #clean the data
    clean_data = DataCleaning.data_pipeline(data_dict)
    
    #preprocess the data
    ##processed_data_RandomForestRegressor = PreProcessData.context_based_pipeline_data(clean_data)
    ##x_train = processed_data_RandomForestRegressor[0]
    ##x_test = processed_data_RandomForestRegressor[1]
    ##y_train = processed_data_RandomForestRegressor[2]
    ##y_test = processed_data_RandomForestRegressor[3]
    
    df_to_process = PreProcessData.context_based_dataFrame(clean_data)
    
    gradient_boosting_df = PreProcessData.gradientBoosting_data(df_to_process)
    gradient_boost_data = PreProcessData.context_based_pipeline_data(gradient_boosting_df)
    x_train = gradient_boost_data[0]
    x_test = gradient_boost_data[1]
    y_train = gradient_boost_data[2]
    y_test = gradient_boost_data[3]
    
    #train the model and evaluet him
    ##context_based_model_RandomForestRegressor = TrainModels.context_base_train_model(x_train,y_train)
    ##TrainModels.evaluate_context_based_model(context_based_model_RandomForestRegressor,x_test,y_test)
    
    #improve the model
    ##improve_RandomForestRegressor_model = TrainModels.context_model_hyperparameters_improvement(x_train,y_train)
    ##TrainModels.evaluate_context_based_model(improve_RandomForestRegressor_model,x_test,y_test)
    
    #trying gradientBoosting model
    context_based_gradientBoosting_model = TrainModels.context_based_gradientBoosting_train_model(x_train, y_train)
    TrainModels.evaluate_context_based_model(context_based_gradientBoosting_model, x_test, y_test)
    improve_gradientBoosting = TrainModels.gradient_boosting_hyperparameters_improvement(x_train,y_train)
    TrainModels.evaluate_context_based_model(improve_gradientBoosting,x_test,y_test)
    
if __name__ == "__main__":
    main()