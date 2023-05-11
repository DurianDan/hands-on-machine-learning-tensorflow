from .cluster_similarity import ClusterSimilarity
from .custom_pipelines import (
    log_pipeline,
    cat_pipeline,
    default_num_pipeline,
    ratio_pipeline)
from sklearn.compose import ColumnTransformer, make_column_selector

cluster_simil = ClusterSimilarity(
    n_clusters=10, gamma=1, random_state=42
)

preprocessing_end_part1=ColumnTransformer([
    ("bedrooms", ratio_pipeline, ['total_bedrooms','total_rooms']),
    ('rooms_per_house', ratio_pipeline, ['total_rooms',"households"]),
    ('people_per_house', ratio_pipeline, ['population',"households"]),
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                           "households", "median_income"]),
    ('geo', cluster_simil, ['latitude','longitude']),
    ('cat', cat_pipeline, make_column_selector(dtype_include=object))], # Ocean Proximity
    remainder=default_num_pipeline # housing_median_age
    )