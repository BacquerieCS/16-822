import numpy as np

import utils


img_path: str = f"data/q1/cube.jpg"

if __name__ == "__main__":
    annotations: np.ndarray = utils.annotate(img_path)
    print(annotations)
