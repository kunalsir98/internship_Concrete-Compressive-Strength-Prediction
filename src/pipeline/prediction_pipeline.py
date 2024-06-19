import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Paths to the preprocessor and model artifacts
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            # Load the preprocessor and model objects
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # Transform the input features using the preprocessor
            data_scaled = preprocessor.transform(features)

            # Predict using the loaded model
            predictions = model.predict(data_scaled)
            return predictions

        except Exception as e:
            logging.info("Exception occurred in prediction")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age):
        self.cement = cement
        self.blast_furnace_slag = blast_furnace_slag
        self.fly_ash = fly_ash
        self.water = water
        self.superplasticizer = superplasticizer
        self.coarse_aggregate = coarse_aggregate
        self.fine_aggregate = fine_aggregate
        self.age = age

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'cement': [self.cement],
                'blast_furnace_slag': [self.blast_furnace_slag],
                'fly_ash': [self.fly_ash],
                'water': [self.water],
                'superplasticizer': [self.superplasticizer],
                'coarse_aggregate': [self.coarse_aggregate],
                'fine_aggregate': [self.fine_aggregate],
                'age': [self.age]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.error('Exception occurred in data transformation process')
            raise CustomException(e, sys)

# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create an instance of CustomData with sample input
    custom_data = CustomData(
        cement=540.0,
        blast_furnace_slag=0.0,
        fly_ash=0.0,
        water=162.0,
        superplasticizer=2.5,
        coarse_aggregate=1040.0,
        fine_aggregate=676.0,
        age=28
    )

    # Convert custom data to DataFrame
    input_df = custom_data.get_data_as_dataframe()

    # Create an instance of PredictPipeline and make predictions
    predict_pipeline = PredictPipeline()
    predictions = predict_pipeline.predict(input_df)

    print(predictions)
