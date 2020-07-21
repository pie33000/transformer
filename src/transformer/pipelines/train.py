from kedro.pipeline import node, Pipeline
from transformer.nodes.train import (
    launch_training,
)

def create_training_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=launch_training,
                inputs="reviews",
                outputs="output",
                name="train",
            ),
        ]
    )