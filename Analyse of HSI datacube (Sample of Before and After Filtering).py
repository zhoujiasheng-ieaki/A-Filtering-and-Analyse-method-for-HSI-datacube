import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split

def Analyse_All_pixels_each_wl_with_3xSD(datacube,title=''):
    aa,bb,cc,dd=datacube.shape
    average_spec_all=np.average(datacube.reshape(aa,bb*cc,dd),axis=1)
        
    fontsize=26
    
    std_1 =np.std(average_spec_all,axis=0)
    std_3=std_1.copy()*3
    aver_of_average_spec_all = np.average(average_spec_all,axis=0)
    
    plt.figure(figsize=(30,12))
    plt.boxplot(datacube.reshape(-1,datacube.shape[-1]))
    #aver_of_average_spec_all = np.average(average_spec_all,axis=0)
    plt.plot(aver_of_average_spec_all,c='k',label='average of average_spec_all')
    plt.plot(aver_of_average_spec_all+std_1[:],c='gray',label='± 1 * sd of average_spec_all')
    plt.plot(aver_of_average_spec_all-std_1[:],c='gray')
    plt.plot(aver_of_average_spec_all+std_3[:],c='r',label='± 3 * sd of average_spec_all')
    plt.plot(aver_of_average_spec_all-std_3[:],c='r')
    plt.xticks(np.arange(0,len(wavelength),20),[f"{i} nm" for i in wavelength[np.arange(0,len(wavelength),20)]],rotation=0,fontsize=fontsize)
    plt.legend(fontsize=fontsize,loc='upper left')
    plt.yticks(fontsize=fontsize)
    if title != '':
        plt.title(title,fontsize=fontsize)
    plt.show()

def Analyse_Principal_image_of_one_sample_use_PCA(datacube, sample_no, n_components, return_clim_list, use_the_clim_list = ''):
    aa,bb,cc,dd=datacube.shape
    average_spec_all=np.average(datacube.reshape(aa,bb*cc,dd),axis=1)
    
    assert sample_no < aa, f'sample_no must be smaller than samples number, get {sample_no} but should be < {aa}.'
    
    dataset = average_spec_all.copy()
    pca=PCA(n_components=n_components)
    pca.fit(dataset)
    feature=pca.transform(dataset)
    loading=pca.components_
    explained_variance_ratio=pca.explained_variance_ratio_*100
    print(feature.shape,loading.shape)
    
    y = np.add.accumulate(pca.explained_variance_ratio_)
    x = np.arange(y.size)+1
    plt.figure(figsize=(20,5))
    plt.plot(x, y, ls='-', marker='o',c='k')
    plt.show()
    
    plt.figure(figsize=(20,5))
    plt.plot(dataset.T)
    plt.title(f"dataset Scpectrum",fontsize=16)
    plt.grid()
    plt.show()
    
    for i in range(feature.shape[-1]):
        plt.figure(figsize=(20,5))
        plt.plot(loading[i])
        plt.title(f"PC{i+1} loading",fontsize=16)
        plt.grid()
        plt.show()
    
    pca_transformed_all_pixel=pca.transform(datacube.reshape(aa*bb*cc,dd)).reshape(aa,bb,cc,-1)    

    if use_the_clim_list == '':
        clim_list=[]
        plt.figure(figsize=(10,8))
        for pc_no in range(4):
            explain_=str(round(pca.explained_variance_ratio_[pc_no]*100,4))
            one_pc_all_pixel=pca_transformed_all_pixel[:,:,:,pc_no]
            img = one_pc_all_pixel[sample_no].copy()   
            plt.subplot(n_components,1,pc_no+1)
            plt.imshow(img,cmap='gray')
            plt.title(f'Transformed image on PC{pc_no+1} loading,\nSample No.{sample_no+1}, Explain {explain_}%')
            clim_list.append(np.array([img.min(),img.max()]))
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(10,8))
        for pc_no in range(4):
            explain_=str(round(pca.explained_variance_ratio_[pc_no]*100,4))
            one_pc_all_pixel=pca_transformed_all_pixel[:,:,:,pc_no]
            img = one_pc_all_pixel[sample_no].copy()   
            plt.subplot(n_components,1,pc_no+1)
            plt.imshow(img,cmap='gray')
            plt.title(f'Transformed image on PC{pc_no+1} loading,\nSample No.{sample_no+1}, Explain {explain_}%')
            plt.clim(use_the_clim_list[pc_no])
        plt.tight_layout()
        plt.show()

    if return_clim_list:
        return clim_list

def Analyse_calculate_LOD_min_max_for_each_pixel(HSI_datacube, c_datacube):
    aa,bb,cc,dd=HSI_datacube.shape
    average_spec_all=np.average(HSI_datacube.reshape(aa,bb*cc,dd),axis=1)

    x_train=average_spec_all.copy()
    y_train=c_datacube.copy()
    
    print("x_train : ",x_train.shape)
    print("y_train : ",y_train.shape)
    
    n_components=3
    pls_model=PLSRegression(n_components=n_components)
    pls_model.fit(x_train,y_train)
    HSI_datacube=HSI_datacube.reshape(aa,bb*cc,dd).copy()
    
    LODmin_list=[]
    LODmax_list=[]
    for i in range(bb*cc):
        print(f'Process pixel {i+1} in all {bb*cc}...',end='\r')
        x_train=HSI_datacube[:,i,:].copy()
        
        SEN = 1/np.linalg.norm(pls_model.coef_.flatten())
        
        varx = np.average(np.var(x_train,axis=0))
        
        varycal = np.var(y_train)
        
        ybarcal=np.average(y_train)
        
        h0min = np.square(np.average(y_train))/np.sum((y_train-np.average(y_train))**2)
        
        T = pls_model.transform(x_train)[:,0].reshape(-1,1)
        
        TTT=np.dot(T.T,T)
        TTT_inv=np.linalg.inv(TTT)
        
        hcal=[]
        for i in range(len(x_train)):
            h=np.dot(np.dot(T[i,0].reshape(1,-1).T, TTT_inv), T[i,0].reshape(1,-1))
            hcal.append(h)
        hcal=np.concatenate(hcal,axis=0).flatten()
        #print('hcal shape : ',hcal.shape)
        
        h0cal = hcal+h0min*(1-((y_train-np.average(y_train))/ybarcal)**2)
        
        h0max=np.max(h0cal)
        
        I=len(x_train)
        
        LODmin=3.3*(((SEN**-2)*varx + (h0min+1/I)*(SEN**-2)*varx + (h0min+1/I)*varycal)**0.5)
        LODmax=3.3*(((SEN**-2)*varx + (h0max+1/I)*(SEN**-2)*varx + (h0max+1/I)*varycal)**0.5)
        
        LODmin_list.append(LODmin)
        LODmax_list.append(LODmax)
    LODmin_list=np.array(LODmin_list).reshape(bb,cc)
    LODmax_list=np.array(LODmax_list).reshape(bb,cc)
    print('\nLODmin_list Shape:',LODmin_list.shape)
    print('LODmax_list Shape:',LODmax_list.shape)
    
    print('LODmin : ', LODmin_list)
    print('LODmax : ', LODmax_list)
    print('Average LODmin : ', np.average(LODmin_list))
    print('Average LODmax : ', np.average(LODmax_list))
    print('SD of LODmin : ', np.std(LODmin_list))
    print('SD of LODmax : ', np.std(LODmax_list))

    return LODmin_list, LODmax_list

def Analyse_Visualize_LOD_min_max_for_each_pixel(LODmin_list, LODmax_list, return_LOD_clim_list,  use_the_LODmin_clim_list = '',  use_the_LODmax_clim_list = ''):
    fontsize=16

    if use_the_LODmin_clim_list != '':
        plt.figure(figsize=(20,8))
        plt.subplot(211)
        plt.imshow(LODmin_list,cmap='gray')
        plt.colorbar(orientation='horizontal')
        plt.title('LODmin of each pixel',fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.clim(use_the_LODmin_clim_list)
    else:
        plt.figure(figsize=(20,8))
        plt.subplot(211)
        plt.imshow(LODmin_list,cmap='gray')
        plt.colorbar(orientation='horizontal')
        plt.title('LODmin of each pixel',fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

    if use_the_LODmax_clim_list != '':
        plt.subplot(212)
        plt.imshow(LODmax_list,cmap='gray')
        plt.colorbar(orientation='horizontal')
        plt.title('LODmax of each pixel',fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.clim(use_the_LODmax_clim_list)
        plt.show()
    else:
        plt.subplot(212)
        plt.imshow(LODmax_list,cmap='gray')
        plt.colorbar(orientation='horizontal')
        plt.title('LODmax of each pixel',fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.show()
    
    plt.figure(figsize=(20,6))
    plt.subplot(131)
    plt.boxplot([LODmin_list.flatten(),LODmax_list.flatten()])
    plt.title('Boxplot of LODmin/LODmax of each pixel',fontsize=fontsize)
    plt.xticks([1,2],['LODmin','LODmax'],fontsize=fontsize)
    plt.yticks(np.arange(LODmin_list.flatten().min(), LODmax_list.flatten().max(), (LODmax_list.flatten().max()-LODmin_list.flatten().min())/10),fontsize=fontsize)

    plt.subplot(132)
    plt.boxplot(LODmin_list.flatten())
    plt.title('Boxplot of LODmin of each pixel',fontsize=fontsize)
    plt.yticks(np.arange(LODmin_list.flatten().min(), LODmin_list.flatten().max(), (LODmin_list.flatten().max()-LODmin_list.flatten().min())/6),fontsize=fontsize)# [round(i,5) for i in np.arange(LODmin_list.flatten().min(), LODmin_list.flatten().max(), ((LODmin_list.flatten().max()-LODmin_list.flatten().min())/6))])
    plt.xticks([1],['LODmin'],fontsize=fontsize)

    plt.subplot(133)
    plt.boxplot(LODmax_list.flatten())
    plt.title('Boxplot of LODmax of each pixel',fontsize=fontsize)
    plt.yticks(np.arange(LODmax_list.flatten().min(), LODmax_list.flatten().max(), (LODmax_list.flatten().max()-LODmax_list.flatten().min())/5),fontsize=fontsize)
    plt.xticks([1],['LODmax'],fontsize=fontsize)
    plt.tight_layout()
    plt.show()

    if return_LOD_clim_list:
        return np.array([LODmin_list.min(), LODmin_list.max()]), np.array([LODmax_list.min(), LODmax_list.max()])



if __name__ == '__main__':
    wavelength=np.load("wavelength_196.npy")[:150]
    HSI_raw_datacube=np.load('trim_overlapped_A_all_float32_all.npy')[:,:,:,:150]
    c_raw_datacube=np.float16(np.load('c_overlapped_all.npy'))
    Filtered_datacube = np.load("Filtered_datacube_MAD.npy")
    print(wavelength.shape)
    print(HSI_raw_datacube.shape) # (n, y, x, wl)
    print(c_raw_datacube.shape)
    print(Filtered_datacube.shape)
    
    Analyse_All_pixels_each_wl_with_3xSD(HSI_raw_datacube, 'Before Filtering')
    Analyse_All_pixels_each_wl_with_3xSD(Filtered_datacube, 'After Filtering')

    clim_list = Analyse_Principal_image_of_one_sample_use_PCA(HSI_raw_datacube, sample_no=200, n_components=5, return_clim_list=True)
    Analyse_Principal_image_of_one_sample_use_PCA(Filtered_datacube, sample_no=200, n_components=5, return_clim_list=False, use_the_clim_list=clim_list)

    LODmin_list, LODmax_list = Analyse_calculate_LOD_min_max_for_each_pixel(HSI_raw_datacube, c_raw_datacube)
    LODmin_clim_list, LODmax_clim_list = Analyse_Visualize_LOD_min_max_for_each_pixel(LODmin_list, LODmax_list, return_LOD_clim_list=True)

    LODmin_list, LODmax_list = Analyse_calculate_LOD_min_max_for_each_pixel(Filtered_datacube, c_raw_datacube)
Analyse_Visualize_LOD_min_max_for_each_pixel(LODmin_list, LODmax_list, return_LOD_clim_list=False,  use_the_LODmin_clim_list = LODmin_clim_list,  use_the_LODmax_clim_list = LODmax_clim_list)
