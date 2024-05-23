import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pywt
from scipy.stats import median_abs_deviation

def get_outlier_maskcube(HSI_raw_datacube):
    aa,bb,cc,dd=HSI_raw_datacube.shape
    average_spec_all=np.average(HSI_raw_datacube.reshape(aa,bb*cc,dd),axis=1)
    
    std_1 =np.std(average_spec_all,axis=0)
    std_3=std_1.copy()*3
    aver_of_average_spec_all = np.average(average_spec_all,axis=0)
    
    up_board = (aver_of_average_spec_all+std_3).copy()
    down_board = (aver_of_average_spec_all-std_3).copy()
    
    outlier_maskcube=np.zeros_like(HSI_raw_datacube,dtype=np.uint8)
    for wl in range(outlier_maskcube.shape[-1]):
        print(f'Process {wl+1} in {dd} ...',end='\r')
        where_up_outlier = np.where(HSI_raw_datacube[:,:,:,wl] > up_board[wl])
        where_down_outlier = np.where(HSI_raw_datacube[:,:,:,wl] < down_board[wl])
        one_mask=np.zeros_like(HSI_raw_datacube[:,:,:,wl],dtype=np.uint8)
        one_mask[where_up_outlier] = 1
        one_mask[where_down_outlier] = 1
        outlier_maskcube[:,:,:,wl] = one_mask.copy()
    print('\nDone.')
    print(f'outlier_maskcube shape: {outlier_maskcube.shape}')
    
    aa,bb,cc,dd=HSI_raw_datacube.shape
    all_pixels_count = aa*bb*cc*dd
    outlier_pixels_count = np.where(outlier_maskcube==1)[0].shape[0]
    print('='*80+f'\nOutlier rate = {100*(outlier_pixels_count/all_pixels_count)}% ({outlier_pixels_count} in all {all_pixels_count}).\n'+'='*80)

    return outlier_maskcube

def Filtering_on_each_wl_img(img, mask, show_detail):
    
    fontsize=14

    mask=cv2.dilate(mask.copy(),np.ones((3,3),np.uint8)).copy()
    
    denoised_img = img.copy()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for i in range(0, len(contours)):
        mask_each=np.zeros_like(mask,np.uint8)
        cv2.drawContours(image=mask_each, contours=contours, contourIdx=i, color=(1), thickness=-1, hierarchy=hierarchy)
        #plt.imshow(mask_each)
        #plt.title(f'mask_each {i+1} in {len(contours)}')
        #plt.show()
        surround=cv2.dilate(mask_each.copy(),np.ones((3,3),np.uint8))-mask_each
        #plt.imshow(surround)
        #plt.title('surround')
        #plt.show()
        # 外れ値の補正
        denoised_img[np.where(mask_each==1)] = np.median(img[np.where(surround==1)])
        #plt.imshow(denoised_img,cmap='gray')
        #plt.title(f'denoised_img {i+1} in {len(contours)}')
        #plt.show()
    
    if show_detail:
        plt.imshow(denoised_img,cmap='gray')
        plt.title('denoised_img')
        #plt.savefig(f'img for paper/5-3.tif',dpi=300,bbox_inches='tight')
        plt.show()
    
    # 2D Wavelet変換によるノイズ除去
    coeffs = pywt.wavedec2(denoised_img, 'db2', level=2)
    
    if show_detail:
        cA, (cH, cV, cD), (cH2, cV2, cD2) = coeffs
    
        plt.figure(figsize=(20,4))
        plt.subplot(241)
        plt.imshow(cA,cmap='gray')
        plt.title('cA',fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        plt.subplot(242)
        plt.imshow(cH,cmap='gray')
        plt.title('cH',fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        plt.subplot(243)
        plt.imshow(cV,cmap='gray')
        plt.title('cV',fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        plt.subplot(244)
        plt.imshow(cD,cmap='gray')
        plt.title('cD',fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        
        plt.subplot(246)
        plt.imshow(cH2,cmap='gray')
        plt.title('cH2',fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        
        plt.subplot(247)
        plt.imshow(cV2,cmap='gray')
        plt.title('cV2',fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        
        plt.subplot(248)
        plt.imshow(cD2,cmap='gray')
        plt.title('cD',fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
    
        plt.tight_layout()
        plt.show()

    detail_coeff = np.concatenate([c.ravel() for c in coeffs[-1]], axis=0)
    threshold = 3 * median_abs_deviation(detail_coeff)
    
    denoised_coeffs = []
    for coeff in coeffs:
        if isinstance(coeff, tuple):
            denoised_detail = [(cv.ravel() * (np.abs(cv.ravel()) <= threshold)).reshape(cv.shape) for cv in coeff]
            denoised_coeffs.append(denoised_detail)
        else:
            denoised_coeffs.append(coeff)
    
    if show_detail:
        cA, (cH, cV, cD), (cH2, cV2, cD2) = denoised_coeffs#[:2]
    
        plt.figure(figsize=(20,4))
        plt.subplot(241)
        plt.imshow(cA,cmap='gray')
        plt.title('cA',fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        #plt.colorbar()
        #plt.show()
        
        plt.subplot(242)
        plt.imshow(cH,cmap='gray')
        plt.title('denoised_cH',fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        plt.subplot(243)
        plt.imshow(cV,cmap='gray')
        plt.title('denoised_cV',fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        plt.subplot(244)
        plt.imshow(cD,cmap='gray')
        plt.title('denoised_cD',fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        
        plt.subplot(246)
        plt.imshow(cH2,cmap='gray')
        plt.title('denoised_cH2',fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        plt.subplot(247)
        plt.imshow(cV2,cmap='gray')
        plt.title('denoised_cV2',fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        plt.subplot(248)
        plt.imshow(cD2,cmap='gray')
        plt.title('denoised_cD',fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
    
        plt.tight_layout()
        plt.show()
    
    
    # 逆変換による画像の再構成
    final_img = pywt.waverec2(denoised_coeffs, 'db2')
    if show_detail:
        plt.imshow(final_img,cmap='gray')
        plt.title('final_img')
        plt.show()
    
        print(f'img mean = {np.average(img)}')
        print(f'final_img mean = {np.average(final_img)}')
        print(f'img sd = {np.std(img)}')
        print(f'final_img sd = {np.std(final_img)}')
        
        plt.boxplot([img.flatten(),final_img.flatten()])
        plt.xticks([1,2],['Before','After'])
        plt.title('Data distribution before / after filtering')
        plt.show()
    return final_img

def Process_Filtering(HSI_raw_datacube):
    if not os.path.exists('Filtered_datacube_MAD.npy'):
        outlier_maskcube = get_outlier_maskcube(HSI_raw_datacube)

        aa,bb,cc,dd = HSI_raw_datacube.shape
        Filtered_datacube = np.zeros_like(HSI_raw_datacube, dtype=np.float32)
        for i in range(aa):
            for wl in range(dd):
                print(f'Process Sample No.{i+1} in {aa}, Wavelength No.{wl+1} in {dd}',end='\r')
                mask = outlier_maskcube[i,:,:,wl].copy()
                img = HSI_raw_datacube[i,:,:,wl].copy()
                final_img = Filtering_on_each_wl_img(img=img, mask=mask, show_detail=False)
                if final_img.shape == (bb,cc):
                    Filtered_datacube[i,:,:,wl] = final_img.copy()
                else:
                    Filtered_datacube[i,:,:,wl] = final_img[:bb,:cc].copy()
        print('\nDone !')
        np.save('Filtered_datacube_MAD.npy',Filtered_datacube,fix_imports=True)
        print(Filtered_datacube.shape)
    else:
        Filtered_datacube = np.load('Filtered_datacube_MAD.npy')
        print(Filtered_datacube.shape)

    return Filtered_datacube



if __name__ == '__main__':
    wavelength=np.load("../wavelength_196.npy")[:150]
    HSI_raw_datacube=np.load('../trim_overlapped_A_all_float32_all.npy')[:,:,:,:150]
    c_raw_datacube=np.float16(np.load('../c_overlapped_all.npy'))
    
    print(wavelength.shape)
    print(HSI_raw_datacube.shape) # (n, y, x, wl)
    print(c_raw_datacube.shape)
    
    Filtered_datacube = Process_Filtering(HSI_raw_datacube)
    print(Filtered_datacube.shape)
