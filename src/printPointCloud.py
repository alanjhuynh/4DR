from pyntcloud import PyntCloud
import numpy as np


cloud = PyntCloud.from_file("test.ply")

cloud.plot("matplotlib")
