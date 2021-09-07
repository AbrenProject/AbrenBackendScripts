import pandas as pd
from sklearn.neighbors import NearestNeighbors

_abs = abs
def customDistance(a, b):
    return _abs(a[0] - b[0]) + _abs(a[1] - b[1]) + _abs(_abs(a[2] - b[2]) - 1) #TODO: Maybe make type values larger  

def getNeighbors():
    starts = pd.read_csv("random_locations10000.csv")
    starts = starts[0:2000]

    typeCol = [x % 2 for x in range(starts.shape[0])]
    idCol = ["i" + str(x) for x in range(starts.shape[0])]
    starts["typeCol"] = typeCol
    starts["id"] = idCol
    starts["name"] = idCol

    knn = NearestNeighbors(n_neighbors = 5, algorithm = 'ball_tree', metric=lambda a,b: customDistance(a,b)) #TODO: Make destination size bigger
    knn.fit(starts.drop(['id', 'name'], axis='columns'))
    _, indices = knn.kneighbors(starts.drop(['id', 'name'], axis='columns'))
    return indices