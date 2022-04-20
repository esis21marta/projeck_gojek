"""Project pipelines."""

from typing import Dict
from kedro.pipeline import Pipeline
from dmp.pipeline import create_pipeline_make_dataset, create_pipeline_feature_engineering
from dmp.pipeline import create_pipeline_modelling, create_pipeline_predicted

def register_pipelines(**kwargs) -> Dict[str, Pipeline]:

    make_dataset_pipeline = create_pipeline_make_dataset()
    feature_engineering_pipeline = create_pipeline_feature_engineering()
    modeling_pipeline = create_pipeline_modelling()
    predicted_pipeline = create_pipeline_predicted()
    pipeline_all = make_dataset_pipeline + feature_engineering_pipeline + modeling_pipeline + predicted_pipeline

    return {
        "make_dataset": make_dataset_pipeline,
        "feature_engineering": feature_engineering_pipeline,
        "modeling": modeling_pipeline,
        "predicted": predicted_pipeline,
        "__default__": pipeline_all,
    }


