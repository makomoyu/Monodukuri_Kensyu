import cv2
import numpy as np
import time
from SaveImage import SaveImage
from RotationClass import ImageRotator
from ORBResultDataClass import ResultData

save_folder_path = r".\check_image"

def preprocess_for_orb(img):
    """
    ORB特徴量用の前処理
    カラー画像・グレースケール画像の両方に対応
    """
    if img is None:
        raise ValueError("画像が None です")

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    gray = clahe.apply(gray)
    SaveImage.save(save_folder_path, "gray.bmp", gray)
    # cv2.imwrite(r".\check_image\gray.bmp", gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray

def calc_orb_score(reference_img, target_img) -> ResultData:

    ref_gray = preprocess_for_orb(reference_img)
    target_gray = preprocess_for_orb(target_img)

    orb = cv2.ORB_create(
        nfeatures=3000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        patchSize=31
    )

    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(target_gray, None)

    if des1 is None or des2 is None:
        return ResultData(
            KeypointsRef=kp1,
            KeypointsTarget=kp2
        )

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 8:
        return ResultData(
            GoodMatches=len(good_matches),
            KeypointsRef=kp1,
            KeypointsTarget=kp2,
            GoodMatchList=good_matches
        )

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None or mask is None:
        return ResultData(
            GoodMatches=len(good_matches),
            KeypointsRef=kp1,
            KeypointsTarget=kp2,
            GoodMatchList=good_matches
        )

    mask_flat = mask.ravel().astype(bool)

    inlier_matches = [
        m for m, ok in zip(good_matches, mask_flat) if ok
    ]

    good_count = len(good_matches)
    inlier_count = len(inlier_matches)

    inlier_ratio = inlier_count / good_count if good_count > 0 else 0.0

    # スコア計算
    inlier_score = min(inlier_count / 120.0, 1.0)
    ratio_score = min(inlier_ratio / 0.75, 1.0)

    score = 0.70 * inlier_score + 0.30 * ratio_score

    if inlier_count < 35:
        score *= inlier_count / 35.0
    if good_count < 50:
        score *= good_count / 50.0

    score = float(max(0.0, min(score, 1.0)))

    return ResultData(
        Score=score,
        GoodMatches=good_count,
        Inliers=inlier_count,
        InlierRatio=inlier_ratio,
        KeypointsRef=kp1,
        KeypointsTarget=kp2,
        GoodMatchList=good_matches,
        InlierMatchList=inlier_matches
    )

def judge_by_orb(target_img, reference_imgs, threshold=0.8):

    if target_img is None:
        raise ValueError("target_img が None です")

    if not isinstance(reference_imgs, list):
        reference_imgs = [reference_imgs]

    best_result = ResultData()
    best_index = None
    best_ref_img = None

    for i, reference_img in enumerate(reference_imgs):
        if reference_img is None:
            continue

        result = calc_orb_score(reference_img, target_img)

        if result.Score > best_result.Score:
            best_result = result
            best_index = i
            best_ref_img = reference_img

    is_ok = best_result.Score >= threshold

    # 描画
    draw_matches = sorted(
        best_result.InlierMatchList,
        key=lambda m: m.distance
    )[:100]

    output = cv2.drawMatches(
        best_ref_img,
        best_result.KeypointsRef,
        target_img,
        best_result.KeypointsTarget,
        draw_matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    label = "OK" if is_ok else "NG"

    text = (
        f"{label} "
        f"score={best_result.Score:.3f} "
        f"good={best_result.GoodMatches} "
        f"inliers={best_result.Inliers} "
        f"ratio={best_result.InlierRatio:.3f}"
    )

    cv2.rectangle(output, (0, 0), (min(output.shape[1], 1400), 60), (0, 0, 0), -1)
    cv2.putText(output,text,(20, 40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0, 255, 255),2)

    result_dict = {
        "判定": "OK" if is_ok else "NG",
        "スコア": round(best_result.Score, 2),
        # "threshold": threshold,
        # "matched_reference_index": best_index,
        # "good_matches": best_result.GoodMatches,
        # "inliers": best_result.Inliers,
        # "inlier_ratio": best_result.InlierRatio
    }

    return result_dict, output

class ORBJudge:

    @staticmethod
    def result(target_color_image, reference_color_image):
        """ORBの結果取得"""
        start_time = time.perf_counter()
        result, result_image = judge_by_orb(
            target_img=target_color_image,
            reference_imgs=reference_color_image,
            threshold=0.8,
        )
        SaveImage.save(save_folder_path, "result.bmp", result_image)
        end_time = time.perf_counter()
        print(f"判定処理時間：{end_time-start_time:.3f}")
        return result
    
    def print(result_dict):
        for key, value in hantei_result.items():
            print(f"{key}:{value}")
        print("="*20)



if __name__ == "__main__":
    start = time.perf_counter()
    test_image_path = r".\test_image\kinoko\a.bmp" # 流動画像（仮）
    kinoko_template_image_path = r".\template_image\kinoko_template.bmp" # きのこの山部分を切り取った画像（テンプレート）

    test_image = cv2.imread(test_image_path)
    kinoko_template_image = cv2.imread(kinoko_template_image_path)

    hantei_result = ORBJudge.result(test_image, kinoko_template_image)
    ORBJudge.print(hantei_result)

    rotated_hantei_result = ORBJudge.result(test_image, ImageRotator.rotate(kinoko_template_image, 180))
    ORBJudge.print(hantei_result)

    end = time.perf_counter()
    print(f"合計処理時間：{end-start:.3f}")


    

