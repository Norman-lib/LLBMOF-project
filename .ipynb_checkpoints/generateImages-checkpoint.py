import numpy as np
def generateImageSequence(width, height, function, sequence=50):

    if (function == "sine"):
        images = []
        for i in range(sequence):
            image = np.zeros((height, width), dtype=np.float64)
            
            x = int(width / 2)
            y = int(height / 2 + height / 8 * np.sin(4 * np.pi * i / sequence))
            image[y-height//4:y+height//4, x-width//4:x+width//4] = 1
            
            images.append(np.abs(image))
        
        return images
        