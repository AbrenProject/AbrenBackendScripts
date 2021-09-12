import pandas as pd
import numpy as np
import json
from sklearn.neighbors import NearestNeighbors

_abs = abs
def customDistance(a, b):
    return _abs(a[0] - b[0]) + _abs(a[1] - b[1]) + _abs(_abs(a[2] - b[2]) - 1) #TODO: Maybe make type values larger  

def getNearestRides(ridesJson, requestsJson):
    (starts, dests) = getLocations(ridesJson, requestsJson)
    result = {
        "startNeighbors": getNeighbors(starts, 3),
        "destinationNeighbors": getNeighbors(dests, 5)
    }

    return result    

def getNeighbors(data, numNeighbors): #TODO: Handle when locations < neighbors
    knn = NearestNeighbors(n_neighbors = numNeighbors, algorithm = 'ball_tree', metric=lambda a,b: customDistance(a,b))
    knn.fit(data.drop(['_id'], axis=1))
    requestsOnly = data.loc[data['typeCol'] == 0]
    _, indices = knn.kneighbors(requestsOnly.drop(['_id'], axis=1))

    loced = data['_id'].loc[indices.reshape(-1)].to_numpy()
    loced = loced.reshape(requestsOnly.shape[0], numNeighbors).tolist()
    result = dict(zip(requestsOnly['_id'], loced)) 
    return result

def getLocations(ridesJson, requestsJson):
    wayPoints = pd.json_normalize(ridesJson, record_path =['route', 'waypointLocations'], meta=['_id'])

    wayPoints = wayPoints.drop(['name'], axis=1)

    ridesDf = pd.json_normalize(ridesJson)
    rideStarts = ridesDf[['_id', 'driverLocation.latitude', 'driverLocation.longitude']]
    rideStarts = rideStarts.rename(columns={'driverLocation.latitude': 'latitude', 'driverLocation.longitude': 'longitude'})
    
    destinationLocations = ridesDf[['_id', 'route.destinationLocation.latitude', 'route.destinationLocation.longitude']]
    destinationLocations = destinationLocations.rename(columns={'route.destinationLocation.latitude': 'latitude', 'route.destinationLocation.longitude': 'longitude'})

    rideDests = pd.concat([wayPoints, destinationLocations])

    requestsDf = pd.json_normalize(requestsJson)
    requestStarts = requestsDf[['_id', 'riderLocation.latitude', 'riderLocation.longitude']]
    requestStarts = requestStarts.rename(columns={'riderLocation.latitude': 'latitude', 'riderLocation.longitude': 'longitude'})

    requestDests = requestsDf[['_id', 'destination.latitude', 'destination.longitude']]
    requestDests = requestDests.rename(columns={'destination.latitude': 'latitude', 'destination.longitude': 'longitude'})

    rideStarts['typeCol'] = np.ones([rideStarts.shape[0], 1])
    rideDests['typeCol'] = np.ones([rideDests.shape[0], 1])

    requestStarts['typeCol'] = np.zeros([requestStarts.shape[0], 1])
    requestDests['typeCol'] = np.zeros([requestDests.shape[0], 1])

    starts = pd.concat([rideStarts, requestStarts]).reset_index(drop=True)
    dests = pd.concat([rideDests, requestDests]).reset_index(drop=True)

    print()
    print(starts)
    print()
    print(dests)

    return (starts, dests)