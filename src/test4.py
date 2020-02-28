import numpy as np

imagePoints1 = np.array([[-1.0], [-1.0], [-1.0]])
N = 0
print(imagePoints1)
tempPoints1 = np.array([[-1.0], [-1.0], [-1.0]])

for i in range(0, 3000, 1):
    if(N == 0):
        imagePoints1[0][N] = float(i*.001)
        imagePoints1[1][N] = float(i*.001)
        imagePoints1[2][N] = float(i*.001)
        N += 1
    else:
        tempPoints1[0][0] = float(i*.001)
        tempPoints1[1][0] = float(i*.001)
        tempPoints1[2][0] = float(i*.001)
        imagePoints1 = np.append(imagePoints1, tempPoints1, axis = 1)

print(imagePoints1)
print(np.size(imagePoints1,1))

f = open("test2.ply", "w")
f.write("ply \n")
f.write("format ascii 1.0\n")
f.write("element vertex " + str(np.size(imagePoints1,1))+"\n")
f.write("property float x \n")
f.write("property float y \n")
f.write("property float z \n")
f.write("property uchar red \n")
f.write("property uchar green \n")
f.write("property uchar blue \n")
f.write("end_header \n")
for i in range(0, 3000, 1):
    f.write(str(imagePoints1[0][i]) + " " + str(imagePoints1[1][i])+ " "  + str(imagePoints1[2][i]) + "\n")
f.close()
