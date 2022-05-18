import numpy as np
import trimesh as tm
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay as dl

#mesh = tm.load("apple.obj", file_type='obj')
mesh = tm.load("Modelos_3D/face_1.obj", file_type='obj')


v = mesh.vertices

f = dl(v[:,(0,1)])

ax = plt.axes(projection='3d')
ax.scatter(v[:,0], v[:,1], v[:,2], alpha = 1, color = "black")
plt.show()

plt.triplot(v[:,0], v[:,1], f.simplices)
plt.show()

mesh = tm.Trimesh(vertices=v, faces=f.simplices)
mesh.show()