import os, sys

sys.path.insert(1, "\\".join(os.path.realpath(__file__).split("\\")[0:-2]))

from super_resolution.services.utils.super_resolution import SuperResolution
import cv2
import os


if __name__ == "__main__":
    
    output_x2 = "C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\media\\output_test\\degraded_X2.png"
    
    output_x4 = "C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\media\\output_test\\degraded_X4.png"
    
    output_folder_x4 = "C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\media\\output_test\\degraded_X4"
    
    image = cv2.imread("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\media\\test\\Beauty.jpg")
    
    blur_image = cv2.GaussianBlur(image, (5, 5), 1.5)
    
    degraded_image = cv2.resize(blur_image, (image.shape[1] // 2, image.shape[0] // 2))
    
    cv2.imwrite(output_x2, degraded_image)
    
    degraded_image = cv2.resize(blur_image, (((image.shape[1] // 4) // 4) * 4, ((image.shape[0] // 4) // 4) * 4))
    
    cv2.imwrite(output_x4, degraded_image)
    
    for i in range(1,6):
        
        image = cv2.imread(f"C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\media\\test\\Beauty\\Beauty ({i}).jpg")
    
        blur_image = cv2.GaussianBlur(image, (5, 5), 1.5)
    
        degraded_image = cv2.resize(blur_image, (((image.shape[1] // 4) // 4) * 4, ((image.shape[0] // 4) // 4) * 4))
    
        cv2.imwrite(os.path.join(output_folder_x4, f"degraded_X4 ({i}).png"), degraded_image)
     
    SuperResolution("BICUBIC_X2", use_bicubic=True, bicubic_scale = 2).apply_super_resolution(
        image_path = output_x2,
        output_path="C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\media\\output_test",
        filename="BICUBIC_X2.png"
    )
    
    SuperResolution("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\Medical_SRGAN_x2_BGR2RGB.pth").apply_super_resolution(
        image_path = output_x2,
        output_path="C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\media\\output_test",
        filename="MEDICAL_SRGAN_X2.png"
    )
    
    SuperResolution("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\Pretrained_SRGAN_x2_BGR2RGB.pth").apply_super_resolution(
        image_path = output_x2,
        output_path="C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\media\\output_test",
        filename="SRGAN_X2.png"
    )
    
    SuperResolution("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\Pretrained_SRResNet_x2_BGR2RGB.pth").apply_super_resolution(
        image_path = output_x2,
        output_path="C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\media\\output_test",
        filename="SRResNet_X2.png"
    )
    
    SuperResolution("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\SRCNN_x2_BGR2YCrCb.pth").apply_super_resolution(
        image_path = output_x2,
        output_path="C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\media\\output_test",
        filename="SRCNN_X2.png"
    )
    
    SuperResolution("BICUBIC_X4", use_bicubic=True, bicubic_scale = 4).apply_super_resolution(
        image_path = output_x4,
        output_path="C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\media\\output_test",
        filename="BICUBIC_X4.png"
    )
    
    SuperResolution("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\Medical_EDVR_x4_BGR2RGB.pth").apply_super_resolution(
        image_path = None,
        folder_path = output_folder_x4,
        output_path="C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\media\\output_test",
        filename="MEDICAL_EDVR_X4.png"
    )
    
    SuperResolution("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\Medical_ESRGAN_x4_BGR2RGB.pth").apply_super_resolution(
        image_path = output_x4,
        output_path="C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\media\\output_test",
        filename="MEDICAL_ESRGAN_X4.png"
    )
    
    SuperResolution("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\Medical_SRGAN_x4_BGR2RGB.pth").apply_super_resolution(
        image_path = output_x4,
        output_path="C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\media\\output_test",
        filename="MEDICAL_SRGAN_X4.png"
    )
    
    SuperResolution("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\Pretrained_EDVR_x4_BGR2RGB.pth").apply_super_resolution(
        image_path = None,
        folder_path = output_folder_x4,
        output_path="C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\media\\output_test",
        filename="EDVR_X4.png"
    )
    
    SuperResolution("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\Pretrained_ESRGAN_x4_BGR2RGB.pth").apply_super_resolution(
        image_path = output_x4,
        output_path="C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\media\\output_test",
        filename="ESRGAN_X4.png"
    )
    
    SuperResolution("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\Pretrained_RRDB_PSNR_x4_BGR2RGB.pth").apply_super_resolution(
        image_path = output_x4,
        output_path="C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\media\\output_test",
        filename="RRDBNet_X4.png"
    )
    
    SuperResolution("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\Pretrained_SRGAN_x4_BGR2RGB.pth").apply_super_resolution(
        image_path = output_x4,
        output_path="C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\media\\output_test",
        filename="SRGAN_X4.png"
    )
    
    SuperResolution("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\Pretrained_SRResNet_x4_BGR2RGB.pth").apply_super_resolution(
        image_path = output_x4,
        output_path="C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\media\\output_test",
        filename="SRResNet_X4.png"
    )
    
    SuperResolution("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\SRCNN_x4_BGR2YCrCb.pth").apply_super_resolution(
        image_path = output_x4,
        output_path="C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\media\\output_test",
        filename="SRCNN_X4.png"
    )

