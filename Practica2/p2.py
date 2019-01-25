#!/usr/bin/env python

import numpy as np
import cv2
import random as rdm

def leeimagen( filename, flagColor ):
    img = cv2.imread( filename, flagColor)
    return img

def showImgWait( imagen, nombre ):
    cv2.namedWindow(nombre, cv2.WINDOW_NORMAL)
    cv2.imshow(nombre, imagen)
    cv2.waitKey(0)
    #cv2.imwrite('./Guardadas/'+nombre+'.png', imagen)
    cv2.destroyAllWindows()
    
def crop_image(img):
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray = img       
    # Nos quedamos con los colores distintos de negro
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)  
    # Obtenemos el contorno
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]       
    # Se calculan las esquinas 
    x, y, w, h = cv2.boundingRect(contours)                               
    # Limitamos la imagen
    return img[y:y+h,x:x+w]
    
# Constantes
sift = cv2.xfeatures2d.SIFT_create( contrastThreshold = 0.06)
surf = cv2.xfeatures2d.SURF_create()
rdm.seed(1)

"""
-----------------
## EJERCICIO 1 ##
-----------------
"""

"""
a) Variar los valores de umbral de la función
"""

yosemite1 = leeimagen('./imagenes/yosemite/Yosemite1.jpg',0)

def ejercicio1a( imagen ):
    
    # SURF
    # Modificamos el umbral Hessiano de SURF
    surf.setHessianThreshold(500)
    
    # detectamos keypoints
    keypoints_sift = sift.detect(imagen, None)
    keypoints_surf = surf.detect(imagen, None)
    
    # Los dibujamos sobre la imagen
    img_with_keypoints = cv2.drawKeypoints(imagen, keypoints_sift, outImage=None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    showImgWait(img_with_keypoints, 'sift')
    img_with_keypoints = cv2.drawKeypoints(imagen, keypoints_surf, outImage=None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    showImgWait(img_with_keypoints, 'surf')
    
    print('Keypoints SIFT =' + str(len(keypoints_sift)))
    print('Keypoints SURF =' + str(len(keypoints_surf)))
    
ejercicio1a(yosemite1)

"""
b) Identificar cuántos puntos se han detectado dentro de cada octava.
"""
# Descompone el valor octave de Sift en octava, capa y escala 
def unpackOctave(num): 
    octave = num & 255;
    layer = (num >> 8) & 255;
    
    if octave >= 128:
        octave = octave | -128
    
    if octave>=0:
        scale = float(1/(1<<octave))
    else:
        scale = float(1<<-octave)
    return(octave+1, layer, scale)

# Genera un color aleatorio
def random_color():
    return [rdm.randint(0,255),rdm.randint(0,255),rdm.randint(0,255)]

# Crea tantos colores como octavas usadas
def create_colors(contador_octavas):
    colors = []
    continuar = True
    # Calcula cuantas octavas se han usado
    for i in range(len(contador_octavas)):
        if contador_octavas[i] == 0 and continuar:
            index = i
            continuar = False
    # Crea index colores
    for i in range(index):
        colors.append(random_color())
    
    return colors

def showFoundKeypoints(array, string ):
    for i in range(len(array)):
        if array[i] == 0:
            continue
        print('Keypoints encontrados en la '+string+' '+str(i)+' = '+str(array[i]))

# --------------  
# --------------  
# --------------  

# Obtiene los puntos que aparecen en cada octava y capa de Sift
def get_octave_sift( imagen ):
    
    # detectamos los keypoints
    keypoints = sift.detect(imagen, None)
    
    contador_octavas = list(np.zeros(10, dtype=int))
    contador_capas = list(np.zeros(10, dtype=int))
    list_octava = []
    list_capa = []
    list_scale = []
    
    # Recorremos los puntos almacenando informacion
    for i in range(len(keypoints)):
        octava, capa, scale = unpackOctave(keypoints[i].octave)
        
        list_octava.append(octava)
        list_capa.append(capa)
        list_scale.append(scale)
        
        # Aumentamos el contador de puntos en octavas
        try: 
            contador_octavas[octava] = contador_octavas[octava]+1
            contador_capas[capa] = contador_capas[capa]+1
        except IndexError:
            print('Mas de 10')
    
        
    return keypoints, list_octava, list_scale, contador_octavas, contador_capas

# Dibuja los círculos de cada punto en la imagen
def draw_circles_sift(imagen, keypoints, octava, scales, contador_octavas):
    # genera tantos colores como octavas ocupadas
    colors = create_colors(contador_octavas)
    
    # Dibuja el círculo con el color de su octava y radio en funcion su sigma
    for i in range(len(keypoints)):
        x = keypoints[i].pt[0]
        y = keypoints[i].pt[1]
        cv2.circle(imagen, center = (round(x),round(y)), radius = round(scales[i]*10), color = colors[octava[i]])
        
    showImgWait(imagen, 'CirclesSift')
   
def ejercicio1bSift(imagen):
    keypoints_sift, octavas_sift, scales, contador_octavas, contador_capas = get_octave_sift(imagen)
    showFoundKeypoints(contador_octavas, 'octava' )
    showFoundKeypoints(contador_capas, 'capa' )
    draw_circles_sift(imagen, keypoints_sift, octavas_sift, scales, contador_octavas)

yosemite1 = leeimagen('./imagenes/yosemite/Yosemite1.jpg',1)
ejercicio1bSift(yosemite1)

# ---------------
# --------------  
# Obtiene los puntos que aparecen en cada octava de Surf
def get_octave_surf( imagen ):
    
    # detectamos los keypoints
    keypoints = surf.detect(imagen, None)
    
    list_octava = []
    contador_octavas = np.zeros(10, dtype=np.float32)
    
    # Recorremos los puntos almacenando informacion
    for i in range(len(keypoints)):
        list_octava.append(keypoints[i].octave)
        
        # Aumentamos el contador de puntos en octavas
        try: 
            contador_octavas[keypoints[i].octave] = contador_octavas[keypoints[i].octave]+1
        except IndexError:
            print('Mas de 10 octavas')
            
    return keypoints, list_octava, contador_octavas

# Dibuja los círculos de cada punto en la imagen
def draw_circles_surf(imagen, keypoints, octava, contador_octavas):
    # genera tantos colores como octavas ocupadas
    colors = create_colors(contador_octavas)
    # Dibuja el círculo con el color de su octava y radio en funcion su sigma
    for i in range(len(keypoints)):
        x = keypoints[i].pt[0]
        y = keypoints[i].pt[1]
        cv2.circle(imagen, center = (round(x),round(y)), radius = 10, color = colors[octava[i]])
        
    showImgWait(imagen, 'CirclesSurf')

def ejercicio1bSurf( imagen ):
    keypoints_surf, octavas_surf, contador_octavas = get_octave_surf(imagen)
    showFoundKeypoints(contador_octavas, 'octava' )
    draw_circles_surf(imagen, keypoints_surf, octavas_surf, contador_octavas)
    
yosemite1 = leeimagen('./imagenes/yosemite/Yosemite1.jpg',1)
ejercicio1bSurf(yosemite1)

"""  
c) Mostrar cómo con el vector de keyPoint extraidos se pueden calcular los descriptores
"""
yosemite1 = leeimagen('./imagenes/yosemite/Yosemite1.jpg',1)
def ejercicio1c(imagen):
    
    # Sift
    key_sift = sift.detect(imagen,None)
    # Obtenemos descriptor
    key2, des_sift = sift.compute(imagen, key_sift)
    
    # Surf
    key_surf = surf.detect(imagen,None)
    # Obtenemos descriptor
    key2, des_surf = sift.compute(imagen, key_surf)
    
    return des_sift, des_surf

ejercicio1c(yosemite1)

"""
-----------------
## EJERCICIO 2 ##
-----------------
"""
"""
a) Usar el detector-descriptor SIFT de OpenCV sobre las imágenes de Yosemite.rar
"""
yosemite1 = leeimagen('./imagenes/yosemite/Yosemite1.jpg', 0)
yosemite2 = leeimagen('./imagenes/yosemite/Yosemite2.jpg', 0)

"""
BruteForce+crossCheck
"""
def BruteForce_CrossCheck(i1, i2, show = True):
    
    # Creamos el objeto de correspondencias
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    
    # Calculamos los descriptores
    kp1, des1 = sift.detectAndCompute(i1,None)
    kp2, des2 = sift.detectAndCompute(i2,None)
    
    # Emparejamos descriptores
    matches = bf.match(des1,des2)
    
    # Ordenar segun mejores
    #matches = sorted(matches, key = lambda x:x.distance)
    rdm.shuffle(matches)
    
    # Draw 100 points
    img3 = cv2.drawMatches(i1, kp1, i2,kp2, matches[:100], outImg=None, flags=2)
    
    if show:
        showImgWait(img3, 'BF-Match')
        return -1
    else:
        return ((kp1,des1),(kp2,des2),matches)
    
BruteForce_CrossCheck(yosemite1,yosemite2)

"""
 Lowe-Average-2NN
"""
def knnMatch(i1, i2, show=True, improve=True):
    
    # Creamos el objeto de correspondencias
    bf = cv2.BFMatcher()
    
    # Calculamos los descriptores
    kp1, des1 = sift.detectAndCompute(i1,None)
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
    
    # Ordenar segun mejores
    # matches = sorted(good, key = lambda x:x[0].distance)
    rdm.shuffle(good)

    if show:
        img4 = cv2.drawMatchesKnn(i1,kp1,i2,kp2,good[:100],outImg=None,flags=2)
        showImgWait(img4, 'KnnMatch')
        return -1
    else:
        return ((kp1,des1),(kp2,des2), good)
    
knnMatch(yosemite1, yosemite2)

"""
---------------------
## EJERCICIO 3 / 4 ##
---------------------
"""

def move_to_center(img, frame):
    
    # Se obtienen las dimensiones de la imagen central
    if len(img.shape) == 3:
        height_frame, width_frame, B = frame.shape
        height_img, width_img, A = img.shape
    else:
        height_frame, width_frame = frame.shape
        height_img, width_img = img.shape
    
    # Se calculan las coordenadas de inicio de la imagen central
    y_displacement = height_frame/2 - height_img/2
    x_displacement = width_frame/2 - width_img/2
    
    # Se crea una matriz 3x3 con los valores a 0 menos la diagonal a 1 y los
    # valores de la traslación
    m0 = np.array([[1, 0, x_displacement], [0, 1, y_displacement], [0, 0, 1]], dtype=np.float32)
    
    # BORDER_CONSTANT para rellenar de negro
    frame = cv2.warpPerspective(img, m0, (width_frame, height_frame), dst=frame, borderMode=cv2.BORDER_CONSTANT)
                  
    return frame, m0

def get_homeography(i1,i2):
    
    # Se calculan las correspondencias
    kpdes1, kpdes2, matches = knnMatch(i1,i2,show=False,improve=True)
    
    kp1, des1 = kpdes1
    kp2, des2 = kpdes2
    
    # Comprobamos tener el mínimo para realizar homografía
    if len(matches)<4:
        return -1
    
    # Cambiamos el formato de los puntos
    src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in matches ]).reshape(-1,1,2)

    # Calculamos homografía
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1)
    
    return M

# Compone el mosaico paso a paso, pegando una imagen detrás de otra
def mosaico_R(imagenes):
    
    # Se calcula el ancho del mosaico como la suma de todos los anchos de las imágenes
    height = sum([i.shape[0] for i in imagenes])
    width = sum([i.shape[1] for i in imagenes])
    
    # Comprobamos gris o RGB
    if len(imagenes[0].shape) == 3:
        frame = np.zeros((height,width,3))
    else:
        frame = np.zeros((height,width))
    
    # Movemos primera imagen al centro
    img, m0 = move_to_center(imagenes[0],frame)
    
    # Calculamos homografías y llevamos imágenes al mosaico una a una
    for i in range(1, len(imagenes)):
        M = get_homeography(imagenes[i],img)
        img = cv2.warpPerspective(imagenes[i], M, (width, height), dst=img, borderMode=cv2.BORDER_TRANSPARENT)
        
    # Recortamos los bordes
    img = crop_image(img)
    return img

# Calcula previamente todas las homografías necesarias para trasladar cada imagen
# al mosaico y por último las traslada 
def mosaico_M(imagenes):
    
    # Se calcula el ancho del mosaico como la suma de todos los anchos de las imágenes
    height = sum([i.shape[0] for i in imagenes])
    width = sum([i.shape[1] for i in imagenes])
    
    # Comprobamos gris o RGB
    if len(imagenes[0].shape) == 3:
        frame = np.zeros((height,width,3))
    else:
        frame = np.zeros((height,width))
    
    # Obtenemos la homografía necesaria para llevar la imagen al centro del marco
    img, m0 = move_to_center(imagenes[0],frame)
    
    homografias = []
    homografias.append(m0)
    
    # Calculamos todas las homeografías necesarias para formar el mosaico
    for i in range(1, len(imagenes)):
        # Homografía de imágenes dos a dos
        M = get_homeography(imagenes[i], imagenes[i-1])
        # Multiplicamos por la homografía del paso anterior
        new_M = np.matmul(m0, M)
        homografias.append(new_M)
        m0 = new_M
    # Trasladamos todas las imágenes al marco
    for i in range(len(imagenes)):
        frame = cv2.warpPerspective(imagenes[i], homografias[i], (width, height), dst=frame, borderMode=cv2.BORDER_TRANSPARENT)
        
    # Recortamos los bordes
    frame = crop_image(frame)
    return frame
  
# Ejercicio 3

def ejercicio3_4( imagenes, ejercicio ):
    im = mosaico_M(imagenes)
    showImgWait(im, 'mosaico_M'+ejercicio)
    im = mosaico_R(imagenes)
    showImgWait(im, 'mosaico_R'+ejercicio)
    
mosaico1 = leeimagen('./imagenes/yosemite_full/yosemite1.jpg',1)
mosaico2 = leeimagen('./imagenes/yosemite_full/yosemite2.jpg',1)
mosaico3 = leeimagen('./imagenes/yosemite_full/yosemite3.jpg',1)

ejercicio3_4([mosaico1,mosaico2,mosaico3], '3')

# Ejercicio 4

mosaico1 = leeimagen('./imagenes/mosaico-1/mosaico002.jpg',1)
mosaico2 = leeimagen('./imagenes/mosaico-1/mosaico003.jpg',1)
mosaico3 = leeimagen('./imagenes/mosaico-1/mosaico004.jpg',1)
mosaico4 = leeimagen('./imagenes/mosaico-1/mosaico008.jpg',1)
mosaico5 = leeimagen('./imagenes/mosaico-1/mosaico011.jpg',1)
mosaico6 = leeimagen('./imagenes/mosaico-1/mosaico009.jpg',1)

ejercicio3_4([mosaico1, mosaico2, mosaico3, mosaico4, mosaico5, mosaico6],'4')





