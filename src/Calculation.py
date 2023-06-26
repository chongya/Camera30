import os
import numpy as np
import pandas as pd
from skimage.transform import resize
from PIL import Image,ExifTags
np.warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

def fVZA(var, p00, p10, p01, p20, p11, p02, p30, p21, p12, p03, p40, p31, p22, p13, p04, p50, p41, p32, p23, p14, p05):
    X = var
    x = X[:,0]
    y = X[:,1]
    z = p00 + p10*x + p01*y + p20*x**2 + p11*x*y + p02*y**2 + p30*x**3 + p21*x**2*y + p12*x*y**2 + p03*y**3 + p40*x**4 + p31*x**3*y + p22*x**2*y**2 + p13*x*y**3 + p04*y**4 + p50*x**5 + p41*x**4*y + p32*x**3*y**2 + p23*x**2*y**3 + p14*x*y**4 + p05*y**5
    return z

def work(task):
    path,direction = task
    
    if 'iPhone' in path:
        p = np.array([ 
        6.56459637e+01, -6.67937318e-03, -1.35464036e-02,  6.68763748e-06,
        4.29502514e-06, -1.00724507e-06, -3.26848475e-09, -8.11099111e-09,
       -1.71803473e-09,  5.99282243e-12,  7.30416448e-13,  4.42314447e-12,
        4.88869077e-13,  7.40806820e-14,  1.00910776e-13, -4.58292591e-17,
       -7.30210969e-16,  1.52974811e-17, -1.21341587e-17, -6.19238342e-18,
        4.80338584e-18])
        height = 4032
        width = 3024
    elif 'Sony' in path:
        p = np.array([ 
        7.17800012e+01, -3.26183075e-03, -8.58007854e-03,  1.91893406e-06,
        2.20982959e-06, -7.75872351e-07, -2.17199187e-10, -3.63230580e-09,
       -3.93459228e-10, -1.33257043e-10, -1.20917015e-13,  1.56603559e-12,
        2.69066750e-13, -1.27805684e-13,  5.64103067e-14,  2.84611545e-17,
       -2.02990056e-16, -3.30089081e-17,  2.06231339e-17,  5.55492020e-18,
       -7.66835059e-19])
        height = 5472
        width = 3648
    p00,p10,p01,p20,p11,p02,p30,p21,p12,p03,p40,p31,p22,p13,p04,p50,pr1,p32,p23,p14,p05 = p  
    C,R = np.meshgrid(np.arange(width),np.arange(height))
    X = np.c_[C.ravel(),R.ravel()].astype(np.float32)
    VZA = fVZA(X,p00,p10,p01,p20,p11,p02,p30,p21,p12,p03,p40,p31,p22,p13,p04,p50,pr1,p32,p23,p14,p05).reshape(R.shape)    
    if direction == 'Upward': VZA = np.flipud(VZA)

    # Masks
    msk575 = np.abs(VZA-57.5) <= 5
    msk0 = np.abs(VZA) <= 5
    msk1 = (VZA>=0) & (VZA<=12.3)
    msk2 = (VZA>=16.7) & (VZA<=28.6)
    msk3 = (VZA>=32.4) & (VZA<=43.4)
    msk4 = (VZA>=47.3) & (VZA<=58.1)
    msk5 = (VZA>=62.3) & (VZA<=74.1)

    # Rings
    thetai = np.array([7,23,38,53,68])
    Wi = np.array([0.041,0.131,0.201,0.290,0.337])
    Wi/=Wi.sum()
    W_i = np.array([0.033,0.097,0.127,0.141,0.102])
    W_i/=W_i.sum()
    W_i/=2

    # Initialize
    ESU = np.array(['A1','A2','A3','A4','B1','B2','B3','B4','C1','C2','C3','C4','D1','D2','D3','D4'])
    LAI = np.array([],dtype=np.float32)
    SEL = np.array([],dtype=np.float32)
    ACF = np.array([],dtype=np.float32)
    DIFN = np.array([],dtype=np.float32)
    MTA = np.array([],dtype=np.float32)
    SEM = np.array([],dtype=np.float32)
    AVGTRANS1 = np.array([],dtype=np.float32)
    AVGTRANS2 = np.array([],dtype=np.float32)
    AVGTRANS3 = np.array([],dtype=np.float32)
    AVGTRANS4 = np.array([],dtype=np.float32)
    AVGTRANS5 = np.array([],dtype=np.float32)
    GAPS1 = np.array([],dtype=np.float32)
    GAPS2 = np.array([],dtype=np.float32)
    GAPS3 = np.array([],dtype=np.float32)
    GAPS4 = np.array([],dtype=np.float32)
    GAPS5 = np.array([],dtype=np.float32)
    FGN = np.array([],dtype=np.float32)
    SEF = np.array([],dtype=np.float32)

    for esu in ESU:
        f = np.array([])
        P1 = np.array([],dtype=np.float32)
        P2 = np.array([],dtype=np.float32)
        P3 = np.array([],dtype=np.float32)
        P4 = np.array([],dtype=np.float32)
        P5 = np.array([],dtype=np.float32)
        fGreen = np.array([],dtype=np.float32)
 
        samples = ['a','b','c','d','e','f','g','h']
        for sample in samples:
            # Read the classification image
            name = path.split('/')[1] + esu + sample
            url = '%s/JPG_Bin/%s.JPG' % (path,name)
            if not os.path.exists(url): url = '%s/JPG_Bin/%s.jpg' % (path,name)
            if os.path.exists(url):
                im = Image.open(url)
                
                # Read the original image and parse metadata
                url0 = '%s/JPG/%s.JPG' % (path,name)
                if not os.path.exists(url0): url0 = '%s/JPG/%s.jpg' % (path,name)
                img = Image.open(url0)
                exif = Image.open(url0)._getexif()
                # Rotate the image
                orientation = 274
                if orientation in exif.keys():
                    if exif[orientation] == 3:
                        im = im.rotate(180,expand=True)
                        img = img.rotate(180,expand=True)
                    elif exif[orientation] == 6:
                        im = im.rotate(270,expand=True)
                        img = img.rotate(270,expand=True)
                    elif exif[orientation] == 8:
                        im = im.rotate(90,expand=True)
                        img = img.rotate(90,expand=True)
                
                # Read oringal image
                img = np.array(img)
                if img.shape[0]!=height or img.shape[1]!=width: img = np.rint(resize(img,(height,width,3))).astype(np.uint8)

                # Read vegetation mask
                msk = np.array(im) < 100
                if msk.shape[0]!=height or msk.shape[1]!=width: msk = np.round(resize(msk,(height,width))).astype(bool)
     
                # Calculate transmittance for the four rings
                p1 = (msk1&msk).sum()/msk1.sum()
                p2 = (msk2&msk).sum()/msk2.sum()
                p3 = (msk3&msk).sum()/msk3.sum()
                p4 = (msk4&msk).sum()/msk4.sum()
                p5 = (msk5&msk).sum()/msk5.sum()
                P1 = np.append(P1, p1)
                P2 = np.append(P2, p2)
                P3 = np.append(P3, p3)
                P4 = np.append(P4, p4)
                P5 = np.append(P5, p5)
                
                # Calculate overall gap fraction of the image
                f = np.concatenate([f,[msk.sum()/msk.size]])

                # Calculate fraction of green tissues
                MSK = (img[:,:,1]>=img[:,:,0]) | (img[:,:,2]>=img[:,:,0])
                N = (~msk&MSK).sum()
                n = (~msk).sum()
                value = np.nan if n == 0 else N/n
                fGreen = np.append(fGreen, value)
            
        if P1.size == 0: 
            ESU = np.delete(ESU,np.where(ESU==esu)[0])
            continue
        
        # Remove outliers
        outliers = (f<(f.mean()-f.std()*3)) | (f>(f.mean()+f.std()*3))
        P1 = P1[~outliers]
        P2 = P2[~outliers]
        P3 = P3[~outliers]
        P4 = P4[~outliers]
        P5 = P5[~outliers]
        fGreen = fGreen[~outliers]
        
        # This is to avoid numerical error
        P1[P1==0] = 1e-5
        P2[P2==0] = 1e-5
        P3[P3==0] = 1e-5
        P4[P4==0] = 1e-5
        P5[P5==0] = 1e-5
        
        # The average probability of light penetration into the canopy
        avgtrans1 = np.mean(P1)
        avgtrans2 = np.mean(P2)
        avgtrans3 = np.mean(P3)
        avgtrans4 = np.mean(P4)
        avgtrans5 = np.mean(P5)
        AVGTRANS1 = np.append(AVGTRANS1, avgtrans1)
        AVGTRANS2 = np.append(AVGTRANS2, avgtrans2)
        AVGTRANS3 = np.append(AVGTRANS3, avgtrans3)
        AVGTRANS4 = np.append(AVGTRANS4, avgtrans4)
        AVGTRANS5 = np.append(AVGTRANS5, avgtrans5)
        
        # The probability of light penetration based on averaging the logarithms of transmittance
        gaps1 = np.exp(np.mean(np.log(P1)))
        gaps2 = np.exp(np.mean(np.log(P2)))
        gaps3 = np.exp(np.mean(np.log(P3)))
        gaps4 = np.exp(np.mean(np.log(P4)))
        gaps5 = np.exp(np.mean(np.log(P5)))
        GAPS1 = np.append(GAPS1, gaps1)
        GAPS2 = np.append(GAPS2, gaps2)
        GAPS3 = np.append(GAPS3, gaps3)
        GAPS4 = np.append(GAPS4, gaps4)
        GAPS5 = np.append(GAPS5, gaps5)
        
        # Leaf area index
        Pij = np.c_[P1,P2,P3,P4,P5]
        Kij = -np.log(Pij)*np.cos(np.radians(thetai))
        Ki_ = np.mean(Kij,0)
        lai = 2*(Ki_*Wi).sum()
        LAI = np.append(LAI, lai)
        
        # Standard error of the leaf area index
        Lj = 2*(Kij*Wi).sum(1)
        sel = np.sqrt(np.mean(Lj**2-lai**2)/Lj.size)
        SEL = np.append(SEL, sel)
        
        # Apparent clumping factor
        Pi_ = np.mean(Pij,0)
        KKi_ = -np.log(Pi_)*np.cos(np.radians(thetai))
        acf = (2*(KKi_*Wi).sum()) / lai
        ACF = np.append(ACF, acf)
        
        # Diffuse non-interceptance
        Gij = Kij / lai
        Gi_ = Gij.mean(0)
        difn = 2*(Gi_*W_i).sum() / 100
        DIFN = np.append(DIFN, difn)

        # Mean tip angle
        m = np.polyfit(np.radians(thetai),Ki_/lai,1)[0]
            
        a0=56.81964
        a1=46.84833
        a2=-64.62133
        a3=-158.69141
        a4=522.06260
        a5=1008.14931
        mta = a0+(a1+(a2+(a3+(a4+a5*m)*m)*m)*m)*m
        if mta > 90: mta = 90
        if mta < 0: mta = 0
        MTA = np.append(MTA, mta)
        
        # Standard error of the mean tip angle
        mj = np.array([np.polyfit(np.radians(thetai),Kij[j]/lai,1)[0] for j in range(Lj.size)])
        mse = mj.std()
        m_ = m - mse if m > 0 else m + mse
        sem = np.abs(mta-(a0+(a1+(a2+(a3+(a4+a5*m_)*m_)*m_)*m_)*m_))
        SEM = np.append(SEM, sem)   

        # Mean and standard error of the green fraction
        FGN = np.append(FGN, np.nanmedian(fGreen))
        SEF = np.append(SEF, np.nanstd(fGreen))
 
    df = pd.DataFrame({'ESU':ESU, 'LAI':LAI, 'SEL':SEL, 'ACF':ACF, 'DIFN':DIFN, 'MTA':MTA, 'SEM':SEM, 'AVGTRANS1':AVGTRANS1, 'AVGTRANS2':AVGTRANS2, 'AVGTRANS3':AVGTRANS3, 'AVGTRANS4':AVGTRANS4, 'AVGTRANS5':AVGTRANS5, 'GAPS1':GAPS1, 'GAPS2':GAPS2, 'GAPS3':GAPS3, 'GAPS4':GAPS4, 'GAPS5':GAPS5, 'FGN':FGN, 'SEF':SEF})
    print(df)

path,direction = '../20200807/iPhone','Upward'
path,direction = '../20200626/Sony','Downward'
work((path,direction))
