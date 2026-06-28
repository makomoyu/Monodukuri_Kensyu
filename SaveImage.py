import os 
import cv2

class SaveImage:
    save_count = 1

    def save(savefolder, savepath, image):
        os.makedirs(savefolder, exist_ok=True)
        cv2.imwrite(os.path.join(savefolder, f"{SaveImage.save_count}_{savepath}"), image)
        SaveImage.save_count += 1