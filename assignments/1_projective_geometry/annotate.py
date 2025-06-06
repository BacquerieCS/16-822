import numpy as np

import utils


feature: str = "tv-perspective"
img_path: str = f"data/q3/{feature}.jpg"

if __name__ == "__main__":
    annotations: np.ndarray = utils.annotate(img_path)
    np.save(f"data/annotation/q3_{feature}.npy", {feature: annotations})
