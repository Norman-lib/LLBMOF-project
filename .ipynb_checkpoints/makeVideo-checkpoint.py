import cv2
import numpy as np

def makeVideo(videoCube, filename, fps=5, asOnes=False):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./Output/'+filename, fourcc, fps, (videoCube.shape[1], videoCube.shape[2]) ,isColor=False)
    ones = videoCube
    if asOnes:
        ones = np.where( videoCube > 0, 1, 0)
    for t in range(videoCube.shape[0]):
        out.write(ones[t, :, :].astype(np.uint8)*255)
    out.release()