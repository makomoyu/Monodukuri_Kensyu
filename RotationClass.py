import cv2

class ImageRotator:
    def rotate(image, angle):
        """
        画像を回転する
        """
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # 回転行列を作成
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # アフィン変換で回転
        rotated = cv2.warpAffine(image, M, (w, h))

        return rotated