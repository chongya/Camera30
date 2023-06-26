import os
from glob import glob
import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2lab
from skimage.morphology import binary_dilation,disk,remove_small_objects
from skimage.segmentation import find_boundaries
from scipy.ndimage import gaussian_filter1d

def mark_peaks(x,r=1,eps=100):
    '''
    Mark peaks for a given 1-d array
    x: 1-d array
    r: moving window radius (moving window size = 2 * r + 1)
    '''
    # Initialize peak flags
    flg = np.full(x.size,False)
    # Judge and mark, item by item
    for i in range(x.size):
        # Define left end of the moving window
        left = max(0,i-r)
        # Define right end of the moving window
        right = min(x.size-1,i+r)
        # Retrive x values within the moving window
        win = x[left:right+1]
        # Is the current item larger than all items within the moving window except for itself?
        criteria = (x[i]>win).sum() == win.size - 1
        # If the criteria is met, then the current item is a peak
        if criteria: 
            flg[i] = True
    flg[x<eps] = False
    return flg

def mark_valleys(x,r=1,eps=100):
    '''
    Mark valleys for a given 1-d array
    x: 1-d array
    r: moving window radius (moving window size = 2 * r + 1)
    '''
    # Initialize peak flags
    flg = np.full(x.size,False)
    # Judge and mark, item by item
    for i in range(x.size):
        # Define left end of the moving window
        left = max(0,i-r)
        # Define right end of the moving window
        right = min(x.size-1,i+r)
        # Retrive x values within the moving window
        win = x[left:right+1]
        # Is the current item larger than all items within the moving window except for itself?
        criteria = (x[i]<win).sum() == win.size - 1
        # If the criteria is met, then the current item is a peak
        if criteria: 
            flg[i] = True
    flg[x<eps] = False
    return flg
    
def work(task):
    file,direction = task

    ## Histogram establishment
    
    # Read image
    img = np.array(Image.open(file))

    if direction == 'Downward':
        # Convert to Lab color space and extract the "a" channel
        a = rgb2lab(img)[:,:,1].astype(np.float32)
        # Calculate vegetation index
        VI = 1-(a-a.min())/(a.max()-a.min())    
    elif direction == 'Upward':
        # Extract blue channel
        B = img[:,:,2].astype(np.float32)
        # Normalize blue channel by removing brightness
        b = B / (img.sum(2).astype(np.float32)+1)
        # Calculate vegetation index
        Bb = B * b
        VI = 1-(Bb-Bb.min())/(Bb.max()-Bb.min())

    # Build histogram
    bins = 200
    interval = 1 / 200
    vi = np.arange(interval/2,1+1e-5,interval)
    frequency = np.histogram(VI.ravel(),bins,range=(0,1))[0]
    # Fill gaps in histogram
    frequency = np.interp(vi,vi[frequency>0],frequency[frequency>0])
    # Smooth histogram
    frequency = gaussian_filter1d(frequency,2)

    ## Peak detection
    
    # Increase search window iteratively
    for r in range(1,bins):
        # Makers of the peaks
        flg_peaks = mark_peaks(frequency,r=r,eps=VI.size*3e-4)
        # Vegetation index values of the peaks
        vi_peaks = vi[flg_peaks]
        # Frequency values of the peaks
        freq_peaks = frequency[flg_peaks]
        # The most prominent one of two peaks
        if vi_peaks.size <= 2: break
    
    ## Identify peaks
    
    # Default is background
    label_peaks = np.zeros(vi_peaks.size,np.uint8)
    for i,vi_peak in enumerate(vi_peaks):
        # Non-blue peaks are not sky
        if direction == 'Upward':
            if np.argmax(img[np.abs(VI-vi_peak)<=interval].mean(0)) != 2:
                label_peaks[i] = 1
        # Non-red peaks are not soil
        if direction == 'Downward':
            if np.argmax(img[np.abs(VI-vi_peak)<=interval].mean(0)) == 1:
                label_peaks[i] = 1
    # Constrain
    label_peaks[vi_peaks<0.6] = 0
    
    ## Thresholding
    
    # Only background peaks case
    if np.all(label_peaks==0):
        # Filling value 
        peakVeg = vi[-1]
        # The last background peak
        peakBg = vi_peaks[label_peaks==0][-1]
        # Line between the background peak and the right corner
        p1 = np.array([peakBg,freq_peaks.max()])
        p2 = np.array([1,0])
        # Distance between the line and the histogram curve
        p3 = np.c_[vi,frequency]
        distance = np.cross(p1-p2,p3-p2) / np.linalg.norm(p1-p2)
        # The point with the largest distance
        thre = vi[distance==distance[vi>peakBg].max()].mean()
        
    # Only vegetation peaks case
    elif np.all(label_peaks==1):
        # Filling value 
        peakBg = vi[0]
        # The first vegetation peak
        peakVeg = vi_peaks[label_peaks==1][0]
        # Line between the background peak and the right corner
        p1 = np.array([0,0])
        p2 = np.array([peakVeg,frequency[vi==peakVeg].item()])
        # Distance between the line and the histogram curve
        p3 = np.c_[vi,frequency]
        distance = np.cross(p1-p2,p3-p2) / np.linalg.norm(p1-p2)
        # The point with the largest distance
        thre = vi[distance==distance[vi<peakVeg].max()].mean()

    # Both background and vegetation peaks case
    else:
        # The last background peak
        peakBg = vi_peaks[label_peaks==0][-1]
        # The first vegetation peak
        peakVeg = vi_peaks[label_peaks==1][0]
        # Valley between the two peaks
        between = (vi>peakBg) & (vi<peakVeg)
        thre = vi[(frequency==frequency[between].min())&between].mean()
        
    ## Saturation correction for thresholding
    
    veg0 = VI >= thre
    if direction == 'Upward':
        fSaturation = (img[~veg0,0].mean()-240) / (255-240)
        if fSaturation < 0: fSaturation = 0
        if fSaturation > 1: fSaturation = 1
        thre = peakBg*fSaturation + thre*(1-fSaturation)
    
    ## Results
    
    # Segment image and postprocess
    veg0 = VI >= thre
    bg = remove_small_objects(~veg0,9)
    veg = remove_small_objects(~bg,100)
    
    # Detect boundaries
    bd0 = find_boundaries(veg)
    bd = binary_dilation(bd0,disk(3))
    img_ = img.copy()
    img_ = np.round(img_.astype(np.float32)/img_.max()*255).astype(np.uint8)
    img_[bd,0] = 255
    img_[bd,1] = 0
    img_[bd,2] = 0
    
    # Save
    Image.fromarray(veg).save(file.replace('/JPG','/JPG_Bin'),quality=100)
    Image.fromarray(img_).save(file.replace('/JPG','/JPG_Edge'),quality=100)


path,direction = '../20200807/iPhone','Upward'
path,direction = '../20200626/Sony','Downward'
os.makedirs(f'{path}/JPG_Bin/',exist_ok=True)
os.makedirs(f'{path}/JPG_Edge/',exist_ok=True)
files = glob(f'{path}/JPG/*.JPG')
files.sort()
for file in files:
    work((file,direction))
