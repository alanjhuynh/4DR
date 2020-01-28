import numpy as np

pixelArray = np.array([[5,5],[10,5],[5,10],[10,10]])
print(pixelArray)
locationArray = np.array([[0,0,0,1],[0,5,0,1],[5,0,0,1],[5,5,0,1]])
print(locationArray)

size1 = 2*np.size(pixelArray, 0)
print(size1)

knownPoints = np.empty([size1,12])


pointer = 0
for pixel in pixelArray:
    newArray = np.append(locationArray[pointer], [0,0,0,0])
    newArray = np.append(newArray,-1*pixel[0] * locationArray[pointer])
    print(newArray)
    knownPoints[2*pointer] = newArray
    newArray = np.append([0,0,0,0], locationArray[pointer])
    newArray = np.append(newArray,-1*pixel[1] * locationArray[pointer])
    print(newArray)
    knownPoints[(2*pointer) + 1] = newArray
    pointer += 1
    
print(knownPoints)
