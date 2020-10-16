from tensorflow.keras import layers

class Postprocess(layers.Layer):
    def __init__(self, n_blocks, n_cells, **kwargs) -> None:
        super().__init__(**kwargs)

    def call(self, input):
        pass
