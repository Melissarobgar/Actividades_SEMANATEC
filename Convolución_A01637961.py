"""
By Benjamin Valdes
code taken from https://github.com/benjaminva/semena-tec-tools-vision/blob/master/Scripts/Ejemplos/simple_conv.py

Modified by Melissa Robles García A01637961
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def processImage(image):
    image = cv2.imread(image)
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    return image

def conv_helper(fragment, kernel):
    """ multiplica 2 matices y devuelve su suma"""

    f_row, f_col = fragment.shape
    k_row, k_col = kernel.shape 
    result = 0.0
    for row in range(f_row):
        for col in range(f_col):
            result += fragment[row,col] *  kernel[row,col]
    return result

def convolution(image, kernel):
    """Aplica una convolucion sin padding (valida) de una dimesion 
    y devuelve la matriz resultante de la operación
    """

    image_row, image_col = image.shape #asigna las dimensiones de la imagen 
    kernel_row, kernel_col = kernel.shape #asigna las dimensiones del filtro
   
    output = np.zeros(image.shape) #matriz donde guarda el resultado
   
    for row in range(image_row):
        for col in range(image_col):
                output[row, col] = conv_helper(image[row:row + kernel_row,col:col + kernel_col],kernel)
                                                                                    
    plt.imshow(output, cmap='gray')
    plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
    plt.show()
 
    
    return output
if __name__ == '__main__':
        # Grayscale Image
    image = processImage('imagen.jpg')

        # Edge Detection Kernel
    kernel = np.ones([3,3])

        # Convolve and Save Output
    output = convolution(image, kernel)
