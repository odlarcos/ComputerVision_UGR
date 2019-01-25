#!/usr/bin/env python

import numpy as np
import cv2
import auxFunc as aux
import math

def leeimagen( filename, flagColor ):
    img = cv2.imread( filename, flagColor)
    return img

def showImgWait( imagen, nombre ):
    cv2.namedWindow(nombre, cv2.WINDOW_NORMAL)
    cv2.imshow(nombre, imagen)
    cv2.waitKey(0)
    #cv2.imwrite('./Guardadas/'+nombre+'.png', imagen)
    cv2.destroyAllWindows()
    
def pintaMismoMarco(vim, TITULO):

    concat = vim[0]
    for i in range(len(vim)-1):
        im = vim[i+1]
        # Insertamos hasta igualar sizes
        while concat.shape[0] > im.shape[0]:
            im = np.insert(im, 0, values=0, axis=0)
        concat = np.concatenate((concat, im), axis=1)
    
    showImgWait(concat, TITULO)

sift = cv2.xfeatures2d.SIFT_create()

"""
# Ejercicio 1
"""

def knnMatch_mask(i1, i2, pts, show=True, improve=True):
    
    # Creamos el objeto de correspondencias
    bf = cv2.BFMatcher()
    
    mask = np.zeros(i1.shape[:2])
    cv2.fillConvexPoly(mask, np.array(pts , dtype=np.int32), color=1)
    
    # Calculamos los descriptores
    kp1, des1 = sift.detectAndCompute(i1,mask=np.array(mask, dtype=np.uint8 ) )
    kp2, des2 = sift.detectAndCompute(i2,None)
    
    # Emparejamos descriptores
    matches = bf.knnMatch(des1, des2, k = 2)
    
    # Aplicar ratio de 0.7 (Lowe)
    if improve:
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append([m])     
    else:
        good = matches
    
    if show:
        img4 = cv2.drawMatchesKnn(i1,kp1,i2,kp2,good[:100],outImg=None,flags=2)
        showImgWait(img4, 'Correspondencias'+str(pts[0][0]))
        return -1
    else:
        return ((kp1,des1),(kp2,des2), good)
    
def ej1():
    
    # 407-408
    i1 = leeimagen('./imagenes/407.png',1)
    i2 = leeimagen('./imagenes/408.png',1)
    
    #a = aux.extractRegion(i1)
    pts = [ (19,289), (366,352), (251,414), (21,347) ]
    
    knnMatch_mask(i1, i2, pts)
    
    # 128 - 130 futbolin
    
    i1 = leeimagen('./imagenes/128.png',1)
    i2 = leeimagen('./imagenes/130.png',1)
    
    pts = [(457,183),(303,306),(395,464),(618,397),(635,208)]
    knnMatch_mask(i1, i2, pts)
    
    # 428 - 429 centralperk
    
    i1 = leeimagen('./imagenes/429.png',1)
    i2 = leeimagen('./imagenes/428.png',1)
    
    pts = [(396,78),(395,137),(577,137),(396,78)]
    
    knnMatch_mask(i1, i2, pts)

ej1()

"""
# Ejercicio 2
"""

def distancia_euclidiana(v1, v2):
    
    return np.linalg.norm(v1-v2)

def similitud(d,q):
    
    return np.dot(d,q) / ( cv2.norm(d, cv2.NORM_L2) * cv2.norm(q, cv2.NORM_L2) )

def leer_imagenes(indices):
    
    imagenes = []
    for i in indices:
        im = leeimagen('./imagenes/'+str(i)+'.png', 1)
        imagenes.append(im)
        
    return imagenes

def calcular_descriptores(im):
    kp, des = sift.detectAndCompute(im, None)
    return des

def calcular_histograma(im, dictionary):
    
    descriptores_imagen = calcular_descriptores(im)
    
    histograma = np.zeros( len(dictionary) )
    for descriptor in descriptores_imagen:
        min_d_euclidea = math.inf
        pos = 0
        for i in range(len(dictionary)):
            d_euclidea = distancia_euclidiana(descriptor, dictionary[i]);
            if d_euclidea < min_d_euclidea:
                min_d_euclidea = d_euclidea
                pos = i
        histograma[pos] = histograma[pos]+1
        
    return histograma

def calcular_histograma2(im, dictionary):
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    descriptores_imagen = calcular_descriptores(im)
    
    histograma = np.zeros( len(dictionary) )  
    matches = bf.match(descriptores_imagen , dictionary)
    
    for m in matches:
        pos = m.trainIdx
        histograma[ pos ] = histograma[ pos ] + 1 
        
    return histograma
    
def inverted_index(histogramas, d):
    
    indice_invertido = []
    for i in range(len(d)): # Para cada palabra
        imagenes = []
        for j in range(len(histogramas)): # Recorremos los histogramas
            if histogramas[j][i] != 0: # Si el histograma contiene esa palabra
                imagenes.append(j)
        indice_invertido.append(imagenes)
        
    return indice_invertido

def imagenes_similares(imagen_pregunta, imagenes):
    
    ac1, labels1, d = aux.loadDictionary('./include/kmeanscenters2000.pkl')
    
    # Calculo todos los histogramas
    histograma_pregunta = calcular_histograma2(imagen_pregunta, d)  
    histogramas = []
    for i in imagenes:
        histograma = calcular_histograma2(i, d)
        histogramas.append(histograma)
        
    # Crear tabla invertida
    # inverted_index(histogramas, d)
    
    # calculo similitudes
    similitudes = []
    for i in range(len(histogramas)):
        sim = similitud(histograma_pregunta, histogramas[i])
        similitudes.append((sim, i))
        
    # Ordeno segun similitud
    similitudes.sort(key=lambda tup: tup[0], reverse = True)
    
    # Escojo las 5 más similares
    imagenes_similares = []
    for i in range(0,5):
        imagenes_similares.append(imagenes[similitudes[i][1]])
        
    return imagenes_similares

def ej2():
    
    imagen_pregunta = leeimagen('./imagenes/15.png', 1)
    showImgWait(imagen_pregunta, 'Pregunta')
    imagenes = leer_imagenes(range(0,440))
    
    similares = imagenes_similares(imagen_pregunta, imagenes)   
    pintaMismoMarco(similares, '5 similares')
    
    imagen_pregunta = leeimagen('./imagenes/292.png', 1)
    showImgWait(imagen_pregunta, 'Pregunta')
    imagenes = leer_imagenes(range(0,440))
    
    similares = imagenes_similares(imagen_pregunta, imagenes)    
    pintaMismoMarco(similares, '5 similares')

ej2()
               
"""
# Ejercicio 3
"""

def best_patches( palabra , titulo ):
    
    descriptors, patches = aux.loadAux('./include/descriptorsAndpatches2000.pkl', True)
    
    distancias = []
    
    distancias = []
    for i in range(len(descriptors)):
        dist = distancia_euclidiana(descriptors[i], palabra);
        distancias.append( (dist, i) )
    
    distancias.sort(key=lambda tup: tup[0])
    
    best_patches = []
    for i in range(10):
        best_patches.append(cv2.cvtColor(patches[ distancias[i][1] ], cv2.COLOR_BGR2GRAY))
        
    pintaMismoMarco(best_patches, 'Mejores patches '+str(titulo))

def ej3():
    ac1, labels1, d = aux.loadDictionary('./include/kmeanscenters2000.pkl')
    best_patches( d[321], 321 );
    best_patches( d[728], 728 );
    
ej3()