#########################
# Computer Vision
# Practica 1
# Oscar David Lopez Arcos
# odlarcos@correo.ugr.es
#########################

#!/usr/bin/env python

import numpy as np
import cv2
from matplotlib import pyplot as plt

#surf = cv2.xfeatures2d.SURF_create()

# --------------------
# Funcion para mostrar imágenes en un mismo canvas
# --------------------

def pintaMismoMarco(vim, TITULO):

    concat = vim[0]
    for i in range(len(vim)-1):
        im = vim[i+1]
        # Insertamos hasta igualar sizes
        while concat.shape[0] > im.shape[0]:
            im = np.insert(im, 0, values=0, axis=0)
        concat = np.concatenate((concat, im), axis=1)
    
    showImagenWait(concat, TITULO)

# --------------------
# Funcion para leer una imagen
# --------------------

def leeimagen( filename, flagColor ):
    img = cv2.imread( filename, flagColor)
    return img

# --------------------
# Funcion para mostrar una imagen
# --------------------

def showImagenWait( imagen, nombre ):
    cv2.namedWindow(nombre, cv2.WINDOW_NORMAL)
    cv2.imshow(nombre, imagen)
    cv2.waitKey(0)
    #cv2.imwrite('./Guardadas/'+nombre+'.png', imagen)
    cv2.destroyAllWindows()
    
# --------------------
# Funcion concatenar títulos (pyplot)
# --------------------
    
def concatenacionTitulos( nfil, ncol, imagenes, titulos):
    
    if len(imagenes) == len(titulos):
        for i in range(len(imagenes)):
            plt.subplot(nfil,ncol,i+1),plt.imshow(imagenes[i]),
            plt.title(titulos[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

# #################
# -----------------
#   EJERCICIO 1
# -----------------
# ##################
        
## a) Convolución con máscara gaussiana 2D

i1 = leeimagen( './imagenes/einstein.bmp', 0)

showImagenWait(i1, 'Original')

# Aplicamos GaussianBlur (ksize a 0 para ser calculado)
blur1 = cv2.GaussianBlur(i1,(0,0),sigmaX=4)
blur2 = cv2.GaussianBlur(i1,(0,0),sigmaX=20)

showImagenWait(blur1, 'image_blur1')
showImagenWait(blur2, 'image_blur2')

imagenes = [i1, blur1, blur2]
titulos = ['Original','Blurred1','Blurred2']

# Concatenamos ambos títulos
concatenacionTitulos(1,3,imagenes,titulos)

## b) Convolución máscara derivadas
def ConvolucionMascaraDerivadas(imagen,dx,dy,ksize,titulo):
    # Obtenemos los kernels de derivadas 
    kernel = cv2.getDerivKernels(dx,dy, ksize=ksize)
    # si ddepth=-1, la imagen destino tendrá igual profundidad que la fuente
    # Pasamos el kernel por la imagen
    i2 = cv2.sepFilter2D(imagen, -1, kernel[0], kernel[1])
    showImagenWait( i2, titulo )

ConvolucionMascaraDerivadas(i1, 1,1, 3, '1MascaraDerivadas')
ConvolucionMascaraDerivadas(i1, 1,1, 5, '2MascaraDerivadas')
ConvolucionMascaraDerivadas(i1, 2,1, 5, '3MascaraDerivadas')
ConvolucionMascaraDerivadas(i1, 0,1, 3, '4MascaraDerivadas')

## c) Laplaciana: derivada tanto en x como y (gradiente)

# (detección de bordes, suele aplicarse a una imagen suavizada
# por un Gaussiano, para reducir su sensibilidad al ruido)

# --------------------
# Funcion para añadir bordes
# --------------------

def addBorders( im, bordertype, bordersize, value):
    row, col= im.shape[:2]
    
    border=cv2.copyMakeBorder(im, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= bordertype, value=value )
    
    return border

# Tipos Bordes: BORDER_CONSTANT, BORDER_DEFAULT, 
# BORDER_REPLICATE, BORDER_WRAP.
    
border1 = addBorders(i1, cv2.BORDER_CONSTANT, 10, 255)
border2 = addBorders(i1, cv2.BORDER_REPLICATE, 10, 0)

def Laplaciana(imagen, sigma, title):
    imagen = cv2.GaussianBlur(imagen,(0,0),sigmaX=sigma)
    # Cuando ksize=1, la Laplacian es calculada filtrando
    # la imagen con una apertura 3x3:
    imagen = cv2.Laplacian(imagen, cv2.CV_64F, ksize=1)
    showImagenWait(imagen, title)

# Define el corte de frecuencias altas que se queda
# Con un sigma más pequeño, más sensible al ruido
Laplaciana(i1, 1, 'LaplacianaOriginal1')
Laplaciana(i1, 3, 'LaplacianaOriginal3')
Laplaciana(border1, 1, 'LaplacianaBordeConstante1')
Laplaciana(border1, 3, 'LaplacianaBordeConstante3')
Laplaciana(border2, 1, 'LaplacianaBordeReplicate1')
Laplaciana(border2, 3, 'LaplacianaBordeReplicate3')

# #################
# -----------------
#   EJERCICIO 2
# -----------------
# ##################

# Si el kernel es simétrico, es separable, pero ser separable no implica ser simétrico
## a) Mascara separable tamaño variable

def MascaraSeparable( nombre_imagen, tam1, indice ):
    i2a = leeimagen( nombre_imagen, 0)
    i2a = addBorders(i2a, cv2.BORDER_REFLECT, 10, 0)
    
    # Pasando sigma negativo, lo calcula a partir de ksize
    kernel1 = cv2.getGaussianKernel(tam1, -1, cv2.CV_32F)
    print(kernel1)
    i2a = cv2.sepFilter2D(i2a, -1, kernel1, kernel1) 
    showImagenWait(i2a, 'MascaraSeparable 2A'+indice)
    
nombre_imagen = 'imagenes/dog.bmp'
MascaraSeparable(nombre_imagen, 5, '1')
MascaraSeparable(nombre_imagen, 20, '2')

# B) Mascara derivada tamaño variable bordes 0

def ConvolucionPrimeraDerivada(nombre_imagen, tam, indice):
    
    i2b = leeimagen(nombre_imagen, 0)
    i2b = addBorders(i2b, cv2.BORDER_CONSTANT, 10, 0)

    kernel = cv2.getDerivKernels(1,1,tam)
    i2b = cv2.sepFilter2D(i2b, -1, kernel[0], kernel[1])
    showImagenWait( i2b, 'Convolucion 2B'+indice )

# Con el tam bajo apenas detecta los cambios de frecuencias
# Con el tam alto detecta muchos más cambios de frecuencia por leves que sean
nombre_imagen = 'imagenes/cat.bmp'
ConvolucionPrimeraDerivada(nombre_imagen, 3,'1')
ConvolucionPrimeraDerivada(nombre_imagen, 9,'2')


# C) 2º derivada 

def ConvolucionSegundaDerivada(nombre_imagen, tam, indice):
    i2c = leeimagen(nombre_imagen, 0)
    kernel = cv2.getDerivKernels(2,2,tam)
    i2c = cv2.sepFilter2D(i2c, -1, kernel[0], kernel[1])
    showImagenWait( i2c, 'Convolucion 2C'+indice )
    
nombre_imagen = 'imagenes/plane.bmp'
ConvolucionSegundaDerivada(nombre_imagen, 3,'1')
ConvolucionSegundaDerivada(nombre_imagen, 9,'2')

# D) Piramide GAUSSIANA

def piramideGaussiana(im, tam):
    
    vim = [im]
    # Tantas veces como niveles quiera mi pirámide
    for i in range(tam):
        # Suavizo
        aux = cv2.GaussianBlur(vim[i],(0,0),sigmaX=2)
        # Reduzco
        aux = cv2.pyrDown(aux)
        # Añado
        vim.append(aux)

    return vim


nombre_imagen = 'imagenes/cat.bmp'
i2d = leeimagen(nombre_imagen, 1)

vim = piramideGaussiana(i2d, 3)
pintaMismoMarco(vim, 'Piramide Gaussiana')

# E) Piramide LAPLACIANA

def ajustar(im):
    rows, cols = im.shape[:2]
    # Compruebo dimensiones impares y añado borde negro para obtener pares
    if rows % 2 == 1:
        im=cv2.copyMakeBorder(im, top=0, bottom=1, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=[0,0,0] )
    if cols % 2 == 1:
        im=cv2.copyMakeBorder(im, top=0, bottom=0, left=0, right=1, borderType=cv2.BORDER_CONSTANT, value=[0,0,0] )
    
    return im

def piramideLaplaciana(im, tam):
    vim = []
    imagenes = [im]
    for i in range(tam):
        # Insertar fila para mantener dimensiones pares
        # de lo contrario, problemas al hacer pyrUp / pyrDown
        imagenes[i] = ajustar(imagenes[i])
        # Suavizo
        aux = cv2.GaussianBlur(imagenes[i],(0,0),sigmaX=2)
        # Reduzco
        aux = cv2.pyrDown(aux)
        # Añado reducida para siguiente iteración
        imagenes.append(aux)
        # Resto la imagen original a la suavizada
        diferencia = cv2.subtract(imagenes[i], cv2.pyrUp(aux))
        # Almaceno la diferencia
        vim.append(diferencia)
    
    vim.append(aux)

    return vim

nombre_imagen = 'imagenes/marilyn.bmp'
i2e = leeimagen(nombre_imagen, 1)
i2e = addBorders(i2e, cv2.BORDER_CONSTANT, 2, [0,0,0])

vim = piramideLaplaciana(i2e, 3)
pintaMismoMarco(vim, 'Piramide Laplaciana')

# #################
# -----------------
#   EJERCICIO 3
# -----------------
# ##################

def imagen_hibridas(high, low, sigma1, sigma2, ksize1, ksize2, indice ,paramSharp = 1 ):
    
    # Aplico el filtro de paso bajo
    blur = cv2.GaussianBlur(low, (ksize1,ksize1), sigma1)
    
    # Aplico el filtro de paso bajo y resto la imagen original
    # (Me quedo con las frecuencias altas)
    blur2 = cv2.GaussianBlur(high, (ksize2,ksize2), sigma2)
    sharp = cv2.multiply(paramSharp,cv2.subtract(high, blur2))
    
    # Monto la imagen híbrida
    mix = cv2.add(sharp, blur)
    
    v = [blur, sharp, mix]
    pintaMismoMarco(v, 'Blur-Sharp-Mix'+indice)
    
    v = piramideGaussiana(mix, 4)
    pintaMismoMarco(v, 'Piramide Gaussiana'+indice)

pareja1a = leeimagen('imagenes/plane.bmp', 0)
pareja1b = leeimagen('imagenes/bird.bmp',0)

imagen_hibridas(pareja1a, pareja1b, sigma1=3, sigma2=11, ksize1=0, ksize2=0, paramSharp=1, indice='1')

pareja2a = leeimagen('imagenes/cat.bmp', 0)
pareja2b = leeimagen('imagenes/dog.bmp',0)

imagen_hibridas(pareja2a, pareja2b, sigma1=7, sigma2=11, ksize1=0, ksize2=0, paramSharp=1, indice='2')

pareja3a = leeimagen('imagenes/marilyn.bmp', 0)
pareja3b = leeimagen('imagenes/einstein.bmp',0)

imagen_hibridas(pareja3a, pareja3b, sigma1=3, sigma2=4, ksize1=0, ksize2=0, paramSharp=2, indice='3')


# {**********************}
# {***** B O N U S ******}
# {**********************}

# ###################
#   EJERCICIO 1 - BONUS
# ###################

# --------------------
# Evalúa la función f
# --------------------
def f(x, sigma):
    return np.exp (-0.5 * ((x*x) / (sigma*sigma)))

# ----------------------
# Construye la máscara 
# ----------------------
def mascaraConvolucion1D (sigma):
    # Calcula el tamaño visto en clase
    tam = 6 * np.round(sigma) + 1
    mascara = []
    # Calcula cada valor de la máscara
    for i in range(tam):
        mascara.append(f(i-np.round(tam/2), sigma))
    # normalizo para que sume 1    
    mascara = np.array(mascara) * (1.0/np.sum(mascara))
    return mascara

mascaraConvolucion1D(3)

# ###################
#   EJERCICIO 2 - BONUS
# ###################

# ----------------------
# Prepara los datos para trabajar la convolución 
# ---------------------- 
def convolucionVectorMascara(im, mask):
    
    mask = list(reversed(mask))
    # Si solo tiene un canal
    if type(im[0]) == int:
        im = calculoConvolucion(im, mask)
    # Si tiene tres canales
    else:
        r,g,b = cv2.split(im)
        r = calculoConvolucion(r, mask)
        g = calculoConvolucion(g, mask)
        b = calculoConvolucion(b, mask)
        
        im = cv2.merge([r,g,b])
    
    return im

# ----------------------
# Prepara los datos para trabajar la convolución 
# ---------------------- 

def calculoConvolucion(im, mask):
    # Calculo ksize
    ksize = int(len(mask)/2)
    # Inserto los bordes
    im = insertarBordesReflejados(im, ksize)
    # Realizo convolución sin incluir los bordes  
    for i in range(ksize, len(im)-ksize-1):
        im[i] = sum(np.multiply(mask, im[(i-ksize):(i+ksize+1)]))/len(mask)
        
    return im
    
# ----------------------
# Inserto bordes reflejados 
# ---------------------- 
def insertarBordesReflejados(im, ksize):
    first = im[0]
    last = im[len(im)-1]
    
    for i in range(ksize):
        im.insert(0, first)
        im.append(last)
        
    return im

# ###################
#   EJERCICIO 3 - BONUS
# ###################

# ###################
#   EJERCICIO 4 - BONUS
# ###################
    
# Código ya implementado dentro de la función imagen_hibridas
    
# ###################
#   EJERCICIO 5 - BONUS
# ###################

def pintaMismoMarco_color(vim, TITULO):

    concat = vim[0]
    for i in range(len(vim)-1):
        im = vim[i+1]
        # Insertamos hasta igualar sizes
        while concat.shape[0] > im.shape[0]:
            im = np.insert(im, 0, values=[0,0,0], axis=0)
        concat = np.concatenate((concat, im), axis=1)
    
    showImagenWait(concat, TITULO)

def imagen_hibridas_color(high, low, sigma1, sigma2, ksize1, ksize2, indice ,paramBlur = 1 ):
    
    # Aplico el filtro de paso bajo
    blur = cv2.GaussianBlur(low, (ksize1,ksize1), sigma1)
    
    # Aplico el filtro de paso bajo y resto la imagen original
    # (Me quedo con las frecuencias altas)
    blur2 = cv2.GaussianBlur(high, (ksize2,ksize2), sigma2)
    sharp = cv2.subtract(high, blur2)
    
    # Monto la imagen híbrida
    mix = cv2.add(sharp, blur)
    
    v = [blur, sharp, mix]
    pintaMismoMarco_color(v, 'Color:Blur-Sharp-Mix'+indice)
    
    v = piramideGaussiana(mix, 4)
    pintaMismoMarco_color(v, 'Color:Piramide Gaussiana'+indice)
    
pareja1a = leeimagen('imagenes/cat.bmp', 1)
pareja1b = leeimagen('imagenes/dog.bmp',1)

imagen_hibridas_color(pareja1a, pareja1b, sigma1=7, sigma2=11, ksize1=0, ksize2=0, indice='1')

pareja1a = leeimagen('imagenes/submarine.bmp', 1)
pareja1b = leeimagen('imagenes/fish.bmp',1)

imagen_hibridas_color(pareja1a, pareja1b, sigma1=9, sigma2=11, ksize1=0, ksize2=0, indice='2')

pareja1a = leeimagen('imagenes/bicycle.bmp', 1)
pareja1b = leeimagen('imagenes/motorcycle.bmp',1)

imagen_hibridas_color(pareja1a, pareja1b, sigma1=9, sigma2=11, ksize1=0, ksize2=0, indice='3')

pareja1a = leeimagen('imagenes/marilyn.bmp', 1)
pareja1b = leeimagen('imagenes/einstein.bmp',1)

imagen_hibridas_color(pareja1a, pareja1b, sigma1=9, sigma2=11, ksize1=0, ksize2=0, indice='4')

pareja1a = leeimagen('imagenes/bird.bmp', 1)
pareja1b = leeimagen('imagenes/plane.bmp',1)

imagen_hibridas_color(pareja1a, pareja1b, sigma1=9, sigma2=11, ksize1=0, ksize2=0, indice='5')
