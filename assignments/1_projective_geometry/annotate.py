import numpy as np


# tv: np.ndarray = utils.annotate("data/q1/tv.jpg")
# pc: np.ndarray = utils.annotate("data/q1/pc.jpg")

annotations = np.load("data/annotation/q1_own.npy", allow_pickle=True).item()
# tv = annotations["tv"][:, :2]
pc = annotations["pc"]

pc = np.array(
    [
        [450.70034449, 15.6598763],
        [804.19763846, 137.7984762],
        [489.45586177, 396.16859139],
        [787.75590385, 683.89894695],
        [450.70034449, 15.6598763],
        [489.45586177, 396.16859139],
        [804.19763846, 137.7984762],
        [787.75590385, 683.89894695],
    ]
)

np.save("data/annotation/q1_own.npy", {"pc": pc})
