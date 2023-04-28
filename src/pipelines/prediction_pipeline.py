import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 Delivery_person_Age: float,
                 Delivery_person_Ratings: float,
                 Vehicle_condition: float,
                 multiple_deliveries: float,
                 distance: float,
                 Weather_conditions: str,
                 Road_traffic_density: str,
                 City: str
                 ):

        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings = Delivery_person_Ratings
        self.Weather_conditions = Weather_conditions
        self.Road_traffic_density = Road_traffic_density
        self.Vehicle_condition = Vehicle_condition
        # self.Type_of_order = Type_of_order,
        self.multiple_deliveries = multiple_deliveries
        self.City = City
        self.distance = distance
       
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                # Delivery_person_Age,Delivery_person_Ratings,Weather_conditions,Road_traffic_density,Vehicle_condition,Type_of_order,multiple_deliveries,City,distance,time_diff_minutes,Time_taken (min)

                'Delivery_person_Age': [self.Delivery_person_Age],
                'Delivery_person_Ratings': [self.Delivery_person_Ratings],
                'Weather_conditions': [self.Weather_conditions],
                'Road_traffic_density': [self.Road_traffic_density],
                'Vehicle_condition': [self.Vehicle_condition],
                # 'Type_of_order': [self.Type_of_order],
                'multiple_deliveries': [self.multiple_deliveries],
                'City': [self.City],
                'distance': [self.distance]
                # 'time_diff_minutes': [self.time_diff_minutes]

            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df

        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e, sys)
