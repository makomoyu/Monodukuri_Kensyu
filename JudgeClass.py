import cv2
import numpy as np
import time
from SaveImage import SaveImage

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

def calc_orb_score(reference_img, target_img):
    """
    2枚の画像をORBで比較してスコアを返す
    """

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
        return {
            "score": 0.0,
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "kp1": kp1,
            "kp2": kp2,
            "good_match_list": [],
            "inlier_match_list": []
        }

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
        return {
            "score": 0.0,
            "good_matches": len(good_matches),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "kp1": kp1,
            "kp2": kp2,
            "good_match_list": good_matches,
            "inlier_match_list": []
        }

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        5.0
    )

    if H is None or mask is None:
        return {
            "score": 0.0,
            "good_matches": len(good_matches),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "kp1": kp1,
            "kp2": kp2,
            "good_match_list": good_matches,
            "inlier_match_list": []
        }

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

    # マッチ数が少ない場合はスコアを下げる
    if inlier_count < 35:
        score *= inlier_count / 35.0

    if good_count < 50:
        score *= good_count / 50.0

    score = float(max(0.0, min(score, 1.0)))

    return {
        "score": score,
        "good_matches": good_count,
        "inliers": inlier_count,
        "inlier_ratio": float(inlier_ratio),
        "kp1": kp1,
        "kp2": kp2,
        "good_match_list": good_matches,
        "inlier_match_list": inlier_matches
    }

def judge_by_orb(target_img, reference_imgs, threshold=0.8, output_path=None):
    """
    カット済み画像同士をORBで比較して判定する汎用関数

    Parameters
    ----------
    target_img : np.ndarray
        判定したい画像

    reference_imgs : np.ndarray or list[np.ndarray]
        正解画像、または正解画像リスト

    threshold : float
        OK判定のしきい値

    output_path : str or None
        マッチング結果画像の保存先
    """

    if target_img is None:
        raise ValueError("target_img が None です")

    if not isinstance(reference_imgs, list):
        reference_imgs = [reference_imgs]

    best = {
        "score": 0.0,
        "reference_index": None,
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "reference_img": None,
        "kp1": None,
        "kp2": None,
        "inlier_match_list": []
    }

    for i, reference_img in enumerate(reference_imgs):
        if reference_img is None:
            continue

        result = calc_orb_score(reference_img, target_img)

        if result["score"] > best["score"]:
            best = {
                "score": result["score"],
                "reference_index": i,
                "good_matches": result["good_matches"],
                "inliers": result["inliers"],
                "inlier_ratio": result["inlier_ratio"],
                "reference_img": reference_img,
                "kp1": result["kp1"],
                "kp2": result["kp2"],
                "inlier_match_list": result["inlier_match_list"]
            }

    is_ok = best["score"] >= threshold

    # 結果画像を保存
    if output_path is not None and best["reference_img"] is not None:
        draw_matches = sorted(
            best["inlier_match_list"],
            key=lambda m: m.distance
        )[:100]

        output = cv2.drawMatches(
            best["reference_img"],
            best["kp1"],
            target_img,
            best["kp2"],
            draw_matches,
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        label = "OK" if is_ok else "NG"
        text = (
            f"{label} "
            f"score={best['score']:.3f} "
            f"good={best['good_matches']} "
            f"inliers={best['inliers']} "
            f"ratio={best['inlier_ratio']:.3f}"
        )

        cv2.rectangle(
            output,
            (0, 0),
            (min(output.shape[1], 1400), 60),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            output,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )

        SaveImage.save(save_folder_path, output_path, output)
        # cv2.imwrite(output_path, output)


    return {
        "判定": "OK" if is_ok else "NG",
        "score": best["score"],
        "threshold": threshold,
        "matched_reference_index": best["reference_index"],
        "good_matches": best["good_matches"],
        "inliers": best["inliers"],
        "inlier_ratio": best["inlier_ratio"]
    }


class ORBJudge:

    @staticmethod
    def result(target_color_image, reference_color_image):
        """ORBの結果取得"""
        start_time = time.perf_counter()
        result = judge_by_orb(
            target_img=target_color_image,
            reference_imgs=reference_color_image,
            threshold=0.8,
            # output_path="result.bmp"
        )
        end_time = time.perf_counter()
        print(f"判定処理時間：{end_time-start_time:.3f}")
        return result



if __name__ == "__main__":

    # test_image_path = r".\kinoko\mushroom_0deg.JPG"
    # test_image_path = "a.bmp"
    test_image_path = r".\kinoko\aaaa.bmp" # 流動画像（仮）
    kinoko_template_image_path = r".\kinoko_template2.bmp" # きのこの山部分を切り取った画像（テンプレート）

    test_image = cv2.imread(test_image_path)
    kinoko_template_image = cv2.imread(kinoko_template_image_path)

    hantei_result = ORBJudge.result(
        target_color_image = test_image,
        reference_color_image= kinoko_template_image
    )
    print(hantei_result)
