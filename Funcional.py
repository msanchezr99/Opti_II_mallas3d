import numpy as np
import trimesh as tm
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay as dl
import random

class ADN:
    def __init__(self,longitud,tasa_mutacion):
        self.target=longitud
        self.tasa_mutacion=tasa_mutacion
        
        
    def crea_ind_inicial(self,malla):
        """
        Recibe la malla: atributo vertices de un objeto mesh.
        """
        lista=[random.randint(0,1) for i in range(self.longitud)]
        vect_bool=[x==1 for x in lista]
        malla[vect_bool]

    def numero_puntos(individuo):
        sum(individuo)

    def a():
        pass


# def main():
#     pass

# if __name__=="__main__":
#     main()


#mesh = tm.load("apple.obj", file_type='obj')
mesh = tm.load("C:/Users/marti/OneDrive/Documentos/personal/Universidad/OptiII/Proyecto/face_1.obj", file_type='obj')

v = mesh.vertices
f = dl(v[:,(0,1)])

# ax = plt.axes(projection='3d')
# ax.scatter(v[:,0], v[:,1], v[:,2], alpha = 1, color = "black")
# plt.show()

# plt.triplot(v[:,0], v[:,1], f.simplices)
# plt.show()

mesh = tm.Trimesh(vertices=v, faces=f.simplices)
#mesh.show()

print(len(v))