from pyntcloud import PyntCloud
import numpy as np


cloud = PyntCloud.from_file("first.ply")

cloud.plot("matplotlib")
