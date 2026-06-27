import cv2
import numpy as np


def resize_to_width(img, target_width=2048):
    """
    入力画像の幅をそろえる。
    撮影サイズが違ってもROI指定を安定させるため。
    """
    h, w = img.shape[:2]

    if w == target_width:
        return img

    scale = target_width / w
    new_h = int(round(h * scale))

    resized = cv2.resize(
        img,
        (target_width, new_h),
        interpolation=cv2.INTER_AREA
    )

    return resized

def crop_kinoko_logo_roi(img):
    """
    きのこの山ロゴがある中央領域を切り出す。
    青パッケージと黄パッケージの両方を含められる広めのROI。
    """
    h, w = img.shape[:2]

    x0 = int(w * 0.22)
    y0 = int(h * 0.25)
    x1 = int(w * 0.84)
    y1 = int(h * 0.58)

    roi = img[y0:y1, x0:x1]

    return roi, (x0, y0)

def preprocess_for_orb(img):
    """
    ORB特徴量用の前処理。
    色差ではなく、形状・エッジ・文字の質感を見る。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray


def extract_orb_features(img, nfeatures=3000):
    """
    ORB特徴点と記述子を抽出する。
    """
    gray = preprocess_for_orb(img)

    orb = cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        patchSize=31
    )

    keypoints, descriptors = orb.detectAndCompute(gray, None)

    return keypoints, descriptors


def match_score_by_orb(reference_roi, target_roi):
    """
    参照ROIと検査ROIの一致スコアを0〜1で返す。
    """
    kp1, des1 = extract_orb_features(reference_roi)
    kp2, des2 = extract_orb_features(target_roi)

    if des1 is None or des2 is None:
        return {
            "score": 0.0,
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0
        }

    if len(kp1) < 20 or len(kp2) < 20:
        return {
            "score": 0.0,
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0
        }

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    knn_matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []

    for pair in knn_matches:
        if len(pair) != 2:
            continue

        m, n = pair

        # Lowe ratio test
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 8:
        return {
            "score": 0.0,
            "good_matches": len(good_matches),
            "inliers": 0,
            "inlier_ratio": 0.0
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
            "inlier_ratio": 0.0
        }

    inliers = int(mask.sum())
    inlier_ratio = inliers / len(good_matches)

    # -----------------------------
    # スコア設計
    # -----------------------------
    # inliers が多いほど「同じロゴらしい」
    # inlier_ratio が高いほど「幾何的にきれいに一致している」
    inlier_score = min(inliers / 120.0, 1.0)
    ratio_score = min(inlier_ratio / 0.75, 1.0)

    score = 0.70 * inlier_score + 0.30 * ratio_score

    # マッチ数が少ない場合は強制的に下げる
    if inliers < 35:
        score *= inliers / 35.0

    if len(good_matches) < 50:
        score *= len(good_matches) / 50.0

    score = float(max(0.0, min(score, 1.0)))

    return {
        "score": score,
        "good_matches": len(good_matches),
        "inliers": inliers,
        "inlier_ratio": float(inlier_ratio)
    }


def build_reference_rois(reference_paths):
    """
    きのこの山の参照画像を複数登録する。
    今回は青パッケージと黄パッケージを両方登録する。
    """
    reference_rois = []

    for path in reference_paths:
        img = cv2.imread(path)

        if img is None:
            raise FileNotFoundError(f"画像が読み込めません: {path}")

        img = resize_to_width(img, target_width=2048)
        roi, _ = crop_kinoko_logo_roi(img)

        reference_rois.append({
            "path": path,
            "roi": roi
        })

    return reference_rois


def judge_kinoko(target_path, reference_rois, threshold=0.8, output_path=None):
    """
    target_path が「きのこの山」かどうか判定する。
    複数参照画像のうち、最も高いスコアを採用する。
    """
    target_img = cv2.imread(target_path)

    if target_img is None:
        raise FileNotFoundError(f"検査画像が読み込めません: {target_path}")

    target_img = resize_to_width(target_img, target_width=2048)
    target_roi, offset = crop_kinoko_logo_roi(target_img)

    best = {
        "score": 0.0,
        "reference": None,
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0
    }

    for ref in reference_rois:
        result = match_score_by_orb(ref["roi"], target_roi)

        if result["score"] > best["score"]:
            best = {
                "score": result["score"],
                "reference": ref["path"],
                "good_matches": result["good_matches"],
                "inliers": result["inliers"],
                "inlier_ratio": result["inlier_ratio"]
            }

    is_kinoko = best["score"] >= threshold

    if output_path is not None:
        output = target_img.copy()

        x0, y0 = offset
        h, w = target_roi.shape[:2]

        color = (0, 0, 255) if is_kinoko else (255, 0, 0)

        cv2.rectangle(
            output,
            (x0, y0),
            (x0 + w, y0 + h),
            color,
            4
        )

        label = "KINOKO OK" if is_kinoko else "NG"
        text = f"{label} score={best['score']:.3f}"

        cv2.putText(
            output,
            text,
            (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            color,
            4
        )

        cv2.imwrite(output_path, output)

    return {
        "判定": "きのこの山" if is_kinoko else "NG",
        "score": best["score"],
        "matched_reference": best["reference"],
        "good_matches": best["good_matches"],
        "inliers": best["inliers"],
        "inlier_ratio": best["inlier_ratio"]
    }


if __name__ == "__main__":

    # この2枚を「きのこの山」の参照画像として登録
    reference_paths = [
        r"kinoko\mushroom_0deg.JPG",
        "kinoko.jpg"
    ]

    reference_rois = build_reference_rois(reference_paths)

    # テスト1: 青パッケージ
    result1 = judge_kinoko(
        target_path=r"kinoko\mushroom_0deg.JPG",
        reference_rois=reference_rois,
        threshold=0.8,
        output_path="result_mushroom.jpg"
    )

    print("mushroom_0deg.JPG")
    print(result1)

    # テスト2: 黄パッケージ
    result2 = judge_kinoko(
        target_path="kinoko.jpg",
        reference_rois=reference_rois,
        threshold=0.8,
        output_path="result_kinoko.jpg"
    )

    print("kinoko.jpg")
    print(result2)


    result3 = judge_kinoko(
        target_path="kinoko_1.jpg",
        reference_rois=reference_rois,
        threshold=0.8,
        output_path="result_kinoko_1.jpg"
    )

    print("kinoko_1.jpg")
    print(result3)
    

    # 別画像を判定したい場合はここに追加
    # result3 = judge_kinoko(
    #     target_path="takenoko.jpg",
    #     reference_rois=reference_rois,
    #     threshold=0.8,
    #     output_path="result_takenoko.jpg"
    # )
    # print("takenoko.jpg")
    # print(result3)