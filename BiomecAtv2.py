# Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import sys



# Calibrando DLT
def dlt_calib(cp3d, cp2d):

    cp3d = np.asarray(cp3d)
    cp2d = np.asarray(cp2d)

    m = np.size(cp3d[:, 0], 0)
    M = np.zeros([m * 2, 11])
    N = np.zeros([m * 2, 1])

    for i in range(m):
        M[i*2,:] = [cp3d[i,0], cp3d[i,1], cp3d[i,2] ,1, 0, 0, 0, 0, -cp2d[i, 0] * cp3d[i, 0], -cp2d[i, 0] * cp3d[i, 1], -cp2d[i, 0] * cp3d[i, 2]]
        M[i*2+1,:] = [0 , 0, 0, 0, cp3d[i,0], cp3d[i,1], cp3d[i,2],1, -cp2d[i,1] * cp3d[i,0],-cp2d[i,1] * cp3d[i,1], -cp2d[i,1] * cp3d[i,2]]
        N[[i*2,i*2+1],0] = cp2d[i, :]

    Mt = M.T
    M1 = inv(Mt.dot(M))
    MN = Mt.dot(N)

    DLT = (M1).dot(MN).T

    return DLT

# Reconstrução 3D
def r3d(DLTs, cc2ds):
    DLTs = np.asarray(DLTs)
    cc2ds = np.asarray(cc2ds)
    
    m = len(DLTs)
    M = np.zeros([2 * m, 3])
    N = np.zeros([2 * m, 1])

    for i in range(m):
        M[i*2,:] = [DLTs[i,0]-DLTs[i,8]*cc2ds[i,0], DLTs[i,1]-DLTs[i,9]*cc2ds[i,0], DLTs[i,2]-DLTs[i,10]*cc2ds[i,0]]
        M[i*2+1,:] = [DLTs[i,4]-DLTs[i,8]*cc2ds[i,1],DLTs[i,5]-DLTs[i,9]*cc2ds[i,1],DLTs[i,6]-DLTs[i,10]*cc2ds[i,1]]
        Np1 = cc2ds[i,:].T
        Np2 = [DLTs[i,3],DLTs[i,7]]
        N[[i*2,i*2+1],0] = Np1 - Np2

    cc3d = inv(M.T.dot(M)).dot((M.T.dot(N)))
    
    return cc3d

# Rodando IDE Python
def rec3d_ide(c1=None, c2=None, ref=None):
    
    # Testando
    if c1 is None:
        dfcp2d_c1 = pd.read_csv('cp2d_c1.txt', delimiter=' ',header=None)
        dfcp2d_c2 = pd.read_csv('cp2d_c2.txt', delimiter=' ',header=None)
        dfcp3d = pd.read_csv('cp3d.txt', delimiter=' ',header=None)
    else:
        dfcp2d_c1 = c1
        dfcp2d_c2 = c2
        dfcp3d = ref
            
    cp2dc1 = np.asarray(dfcp2d_c1)
    cp2dc2 = np.asarray(dfcp2d_c2)
    cp3d = np.asarray(dfcp3d)
    
    dltc1 = dlt_calib(cp3d, cp2dc1)
    dltc2 = dlt_calib(cp3d, cp2dc2)
    
    DLTs = np.append(dltc1, dltc2, axis=0)
    
    cc3d = np.zeros([len(cp2dc1), 3])
    for i in range(len(cp2dc1)):
        cc2ds = np.append([cp2dc1[i, :]], [cp2dc2[i, :]], axis=0)
        cc3d[i, :] = r3d(DLTs, cc2ds).T
    
    return cc3d


def cart2sph(x,y,z):
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r


# Rodando CMD
if __name__ == '__main__':

    # Parametros das cameras
    resolutionx = int(720/2) 
    resolutiony = int(220/2) 
    freq = 120
    
    # Carregando os arquivos das cameras

    # Camera 1
    bola1 = pd.read_csv(str(sys.argv[1]), sep='\s+', header=None, decimal='.')
    bola1[1] = bola1[1] - -resolutionx
    bola1[2] = -1 * (bola1[2] - resolutiony) 
    bola1 = np.asarray(bola1[[1,2]])
    bola1b = bola1
    
    # Camera 2
    bola2 = pd.read_csv(str(sys.argv[2]), sep='\s+', header=None, decimal='.')
    bola2[1] = bola2[1] - -resolutionx
    bola2[2] = -1 * (bola2[2] - resolutiony) 
    bola2 = np.asarray(bola2[[1,2]])
    bola2b = bola2
    
    idx = np.asarray(list(range(len(bola1b))))
    diffball = abs(np.diff(bola1b[:,0])) > 5
    diffball = np.insert(diffball, 0, False)
    phitball = idx[diffball][0]
    idxbefore = idx[0:phitball-2]
    idxafter = idx[phitball+1::]
    idxcimpact = idx[phitball-2:phitball+1]

    print(f'Frame de impacto = {phitball}')
    print(f'Frames de impacto critico  = {idxcimpact}')

    plt.close('all')
    plt.subplot(2,1,1)
    plt.grid(True)
    plt.plot(bola1[:,0],bola1[:,1],'o')
    plt.xlabel('CAM 1 - Cordenada X')
    plt.ylabel('CAM 1 - Cordenada Y')
    resx = 2 * resolutionx
    resy = 2 * resolutiony
    plt.title(f'Cordenadas dos pixels(resolution = {resx} X {resy})')
    
    plt.subplot(2,1,2)
    plt.plot(bola2[:,0],bola2[:,1],'o')
    plt.xlabel('CAM 2 - Cordenada X')
    plt.ylabel('CAM 2 - Cordeada Y')
    plt.grid(True)
    
    # Carregar arquivos de calibração
    datcal_c1 = np.asarray(pd.read_csv(str(sys.argv[3]), sep='\s+', header=None))
    datcal_c1[:, 0] = datcal_c1[:, 0] - -resolutionx
    datcal_c1[:, 1] = -1 * (datcal_c1[:, 1] - resolutiony) 
    
    datcal_c2 = np.asarray(pd.read_csv(str(sys.argv[4]), sep='\s+', header=None))
    datcal_c2[:, 0] = datcal_c2[:, 0] - -resolutionx
    datcal_c2[:, 1] = -1 * (datcal_c2[:, 1] - resolutiony) 

    
    ref = np.asarray(pd.read_csv(sys.argv[5], sep='\s+', header=None))
    ref = ref[:,1:]
    
    
    dltc1 = dlt_calib(ref, datcal_c1)
    dltc2 = dlt_calib(ref, datcal_c2)
    dlts = np.append(dltc1, dltc2, axis=0)
    
    cc3d = np.zeros([len(bola1), 3])
    
    for i in range(len(bola1)):
        cc2ds = np.append([bola1[i, :]], [bola2[i, :]], axis=0)
        cc3d[i, :] = r3d(dlts, cc2ds).T
    
    cc3df = cc3d[idxafter,:]
    coefsx = np.polyfit(idxafter, cc3df[:,0], 1)
    coefsy = np.polyfit(idxafter, cc3df[:,1], 1)
    coefsz = np.polyfit(idxafter, cc3df[:,2], 2)
    
    cc3df[:,0] = coefsx[0] * idxafter + coefsx[1]
    cc3df[:,1] = coefsy[0] * idxafter + coefsy[1]
    cc3df[:,2] = coefsz[0] * idxafter**2 + coefsz[1] * idxafter + coefsz[2]
    
    vels = (np.sqrt(np.sum((np.diff(cc3df, axis=0)**2), axis=1))) / (1/freq) * 3.6
    print(f'Velocidade = {vels}')

    vsaida = cc3df[-1:,:] - cc3d[0,:]

    azimuth, elevation, r = cart2sph(vsaida[0][0], vsaida[0][1], vsaida[0][2])
    pi = np.pi
    azi = azimuth * 180/pi
    elev = elevation * 180/pi
    
    print(f'Angulos: azimuth = {azi}; elevacao = {elev}')
   
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d') 
    ax2.plot3D(cc3d[:,0], cc3d[:,1], cc3d[:,2], 'ro', markersize=10)
    ax2.plot3D(ref[:,0], ref[:,1], ref[:,2], 'b.')
    ax2.plot3D(cc3df[:,0], cc3df[:,1], cc3df[:,2], 'k-o')
    ax2.plot3D([cc3d[0,0],cc3d[0,0]], [cc3d[0,1],cc3d[0,1]], [cc3d[0,2],cc3d[0,2]], 'g.', markersize=10)
    
    ax2.set_zlabel('Z [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_xlabel('X [m]')
    
    # print(cc3df)
    distvet = np.sqrt(np.sum((cc3df[-1,:] - cc3df[0,:])**2))
    
    velmed = distvet / (len(cc3df) * (1/freq)) * 3.6
    plt.title(f'Velocidade (Max = {np.round(max(vels),2)} km/h ; Mean = {np.round(velmed)}); Angles (azi = {np.round(azi,1)}, elev. = {np.round(elev,1)})')
    plt.show()

    # import pdb; pdb.set_trace() 
    resultado = list(np.append(vels, [azi, elev, velmed]))
    np.savetxt(str(sys.argv[6])+'_result.txt', resultado, fmt='%.10f')
    print(f'Velocidade Media= {velmed}')
    print('\n')
    
    # with open(str(sys.argv[6])+'_res.txt', 'w') as output:
    #     output.write(str(resultado))
   
    np.savetxt(str(sys.argv[6])+'.3d', cc3d, fmt='%.10f')
    np.savetxt(str(sys.argv[6])+'_filt.3d', cc3df, fmt='%.10f')

    # np.savetxt(str(sys.argv[6])+'.txt', resultado, fmt='%.10f')

    np.savetxt(str(sys.argv[6])+'.3d', cc3d, fmt='%.10f')