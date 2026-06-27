import cv2


def get_resize_image(image, scale_x, scale_y):
    scale_x = 1.62
    scale_y = 1.60

    new_w = int(round(image.shape[1] * scale_x))
    new_h = int(round(image.shape[0] * scale_y))

    resized_template = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )
    return resized_template


test_image_path = r".\kinoko.jpg"
test_image_path = r".\kinoko\mushroom_0deg.JPG"
kinoko_template_image_path = r".\kinoko_template.bmp"
takenoko_template_image_path = r""


originai_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
kinoko_template = cv2.imread(kinoko_template_image_path, cv2.IMREAD_GRAYSCALE)

kinoko_template = get_resize_image(
    kinoko_template,
    scale_x = 1.62,
    scale_y = 1.60)

_, originai_image = cv2.threshold(originai_image, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, kinoko_template = cv2.threshold(kinoko_template, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)



takenoko_template = cv2.imread(takenoko_template_image_path)
cv2.imwrite("check_origin.bmp", originai_image)



class TemplateMatch:

    _kinoko_result = None
    _takenoko_result = None

    class kinoko:
        def result(image):
            TemplateMatch._kinoko_result = cv2.matchTemplate(
                image,
                kinoko_template,
                cv2.TM_CCOEFF_NORMED
            )
            return TemplateMatch._kinoko_result
        
        def score(image):
            min_value, max_value, min_location, max_location = cv2.minMaxLoc(TemplateMatch.kinoko.result(image))
            return max_value

        def location(image):
            min_value, max_value, min_location, max_location = cv2.minMaxLoc(TemplateMatch.kinoko.result(image))
            return max_location

    class takenoko:
        def result():
            TemplateMatch._takenoko_result = cv2.matchTemplate(
                originai_image,
                takenoko_template,
                cv2.TM_CCOEFF_NORMED
            )
            return TemplateMatch._takenoko_result
        
        def score(image):
            min_value, max_value, min_location, max_location = cv2.minMaxLoc(TemplateMatch.takenoko.result(image))
            return max_value

        def location(image):
            min_value, max_value, min_location, max_location = cv2.minMaxLoc(TemplateMatch.takenoko.result(image))
            return max_location
        
        
    

if __name__ == "__main__":
    print(f"最大相関値：{TemplateMatch.kinoko.score(originai_image)}")
    max_lacotion = TemplateMatch.kinoko.location(originai_image)
    x,y = max_lacotion
    h,w = kinoko_template.shape[:2]

    max_cut_image = originai_image[y:y+h, x:x+w]
    cv2.imwrite("result_image.bmp", max_cut_image)
    







