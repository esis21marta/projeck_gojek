"""
This is a boilerplate pipeline
generated using Kedro 0.18.0
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_booking_df, clean_participant_df, merge_dataset, create_target
from .nodes import driver_distance_to_pickup,driver_distance_to_pickup_with_geopy,hour_of_day,driver_historical_completed_bookings_train
from .nodes import model_building, driver_historical_completed_bookings_test, choose_best_driver

def create_pipeline_make_dataset(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_booking_df,
                inputs=["booking_log","parameters"],
                outputs="booking_log_clean",
                name="booking_log_load",
            ),
            node(
                func=clean_participant_df,
                inputs="participant_log",
                outputs="participant_log_clean",
                name="participant_log_load",
            ),
            node(
                func=merge_dataset,
                inputs=["participant_log_clean", "booking_log_clean","parameters"],
                outputs="dataset_raw",
                name="merge_dataset",
            ),
            node(
                func=create_target,
                inputs=["dataset_raw", "parameters"],
                outputs="dataset",
                name="final_dataset",
            ),
        ]
    )

def create_pipeline_feature_engineering(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=driver_distance_to_pickup,
                inputs="dataset",
                outputs="dataset_distance",
                name="driver_distance_to_pickup_def",
            ),
            node(
                func=driver_distance_to_pickup_with_geopy,
                inputs="dataset_distance",
                outputs="dataset_distance_geopy",
                name="driver_distance_to_pickup_with_geopy_def",
            ),
            node(
                func=hour_of_day,
                inputs="dataset_distance_geopy",
                outputs="dataset_hours",
                name="hour_of_day_def",
            ),
            node(
                func=driver_historical_completed_bookings_train,
                inputs=["dataset_hours", "parameters"],
                outputs="transformed_dataset",
                name="transformed_dataset_def",
            ),
        ]
    )


def create_pipeline_modelling(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_building,
                inputs=["transformed_dataset","parameters"],
                outputs=["model_pickle","metrics_auc"],
                name="training_model",
            ),
        ]
    )

def create_pipeline_predicted(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=driver_distance_to_pickup,
                inputs="test_data",
                outputs="test_data_distance",
                name="driver_distance_to_pickup_test",
            ),
            node(
                func=driver_distance_to_pickup_with_geopy,
                inputs="test_data_distance",
                outputs="test_data_distance_geopy",
                name="driver_distance_to_pickup_with_geopy_test",
            ),
            node(
                func=hour_of_day,
                inputs="test_data_distance_geopy",
                outputs="test_data_hours",
                name="hour_of_day_test",
            ),
            node(
                func=driver_historical_completed_bookings_test,
                inputs=["test_data_hours", "transformed_dataset"],
                outputs="transformed_test_data",
                name="transformed_dataset_test",
            ),
            node(
                func=choose_best_driver,
                inputs=["transformed_test_data","model_pickle"],
                outputs="results",
                name="selected_drivers",
            ),
        ]
    )
