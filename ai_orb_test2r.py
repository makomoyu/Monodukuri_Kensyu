import cv2
import numpy as np
import os


# ==========================================================
# 設定
# ==========================================================

REFERENCE_IMAGE_PATH = r"kinoko\mushroom_0deg.JPG"   # 青いパッケージ
TARGET_IMAGE_PATH = "kinoko.jpg"             # 黄色いパッケージ

OUTPUT_REF_ROI_BOX = "01_ref_roi_box.jpg"
OUTPUT_TARGET_ROI_BOX = "02_target_roi_box.jpg"
OUTPUT_REF_ROI_CROP = "03_ref_roi_crop.jpg"
OUTPUT_TARGET_ROI_CROP = "04_target_roi_crop.jpg"
OUTPUT_GOOD_MATCHES = "05_good_matches.jpg"
OUTPUT_INLIER_MATCHES = "06_inlier_matches_score.jpg"


# ==========================================================
# 画像サイズをそろえる
# ==========================================================

def resize_to_width(img, target_width=2048):
    """
    画像の幅を target_width に揃える。
    撮影画像のサイズが少し違ってもROIが安定するようにする。
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


# ==========================================================
# きのこの山ロゴ周辺ROIを切り出す
# ==========================================================

def crop_kinoko_logo_roi(img):
    """
    きのこの山ロゴがある中央付近を広めに切り出す。
    青パッケージ・黄パッケージの両方に対応する範囲。
    """
    h, w = img.shape[:2]

    x0 = int(w * 0.22)
    y0 = int(h * 0.25)
    x1 = int(w * 0.84)
    y1 = int(h * 0.58)

    roi = img[y0:y1, x0:x1]

    return roi, (x0, y0, x1, y1)


# ==========================================================
# ORB用前処理
# ==========================================================

def preprocess_for_orb(img):
    """
    ORB特徴点抽出用の前処理。
    色ではなく形・エッジ・文字の模様で比較する。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # コントラスト補正
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    gray = clahe.apply(gray)

    # ノイズを少し減らす
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray


# ==========================================================
# ORBマッチングして可視化画像を作る
# ==========================================================

def orb_match_visualize(ref_roi, target_roi):
    """
    2つのROI画像をORBで比較し、
    good matches画像とinlier matches画像を作成する。
    """

    ref_gray = preprocess_for_orb(ref_roi)
    target_gray = preprocess_for_orb(target_roi)

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
        raise RuntimeError("特徴点が十分に見つかりませんでした。")

    # Hamming距離でORB特徴量を比較
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # k=2で近い候補を2つ取る
    knn_matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []

    # Lowe ratio test
    for pair in knn_matches:
        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # RANSACで幾何的に正しい一致だけ残す
    inlier_matches = []
    H = None
    mask = None

    if len(good_matches) >= 8:
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

        if mask is not None:
            mask_flat = mask.ravel().astype(bool)
            inlier_matches = [
                m for m, ok in zip(good_matches, mask_flat) if ok
            ]

    good_count = len(good_matches)
    inlier_count = len(inlier_matches)

    if good_count > 0:
        inlier_ratio = inlier_count / good_count
    else:
        inlier_ratio = 0.0

    # ======================================================
    # スコア計算
    # ======================================================

    inlier_score = min(inlier_count / 120.0, 1.0)
    ratio_score = min(inlier_ratio / 0.75, 1.0)

    score = 0.70 * inlier_score + 0.30 * ratio_score

    # マッチ数が少ない場合はスコアを落とす
    if inlier_count < 35:
        score *= inlier_count / 35.0

    if good_count < 50:
        score *= good_count / 50.0

    score = float(max(0.0, min(score, 1.0)))

    # ======================================================
    # good matches 可視化
    # ======================================================

    good_sorted = sorted(good_matches, key=lambda m: m.distance)[:80]

    good_vis = cv2.drawMatches(
        ref_roi,
        kp1,
        target_roi,
        kp2,
        good_sorted,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # ======================================================
    # inlier matches 可視化
    # ======================================================

    inlier_sorted = sorted(inlier_matches, key=lambda m: m.distance)[:80]

    inlier_vis = cv2.drawMatches(
        ref_roi,
        kp1,
        target_roi,
        kp2,
        inlier_sorted,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # ======================================================
    # スコア文字を描画
    # ======================================================

    text = (
        f"score={score:.3f}  "
        f"good={good_count}  "
        f"inliers={inlier_count}  "
        f"ratio={inlier_ratio:.3f}"
    )

    for vis in [good_vis, inlier_vis]:
        cv2.rectangle(
            vis,
            (0, 0),
            (min(vis.shape[1], 1200), 60),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            vis,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 255),
            2
        )

    return {
        "score": score,
        "good_matches": good_count,
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "good_vis": good_vis,
        "inlier_vis": inlier_vis
    }


# ==========================================================
# メイン処理
# ==========================================================

def main():

    ref_img = cv2.imread(REFERENCE_IMAGE_PATH)
    target_img = cv2.imread(TARGET_IMAGE_PATH)

    if ref_img is None:
        raise FileNotFoundError(f"参照画像が読み込めません: {REFERENCE_IMAGE_PATH}")

    if target_img is None:
        raise FileNotFoundError(f"検査画像が読み込めません: {TARGET_IMAGE_PATH}")

    # 幅を揃える
    ref_img = resize_to_width(ref_img, target_width=2048)
    target_img = resize_to_width(target_img, target_width=2048)

    # ROI切り出し
    ref_roi, ref_box = crop_kinoko_logo_roi(ref_img)
    target_roi, target_box = crop_kinoko_logo_roi(target_img)

    # ROI枠付き画像を作る
    ref_roi_box_img = ref_img.copy()
    target_roi_box_img = target_img.copy()

    for img, box, label in [
        (ref_roi_box_img, ref_box, "Reference ROI"),
        (target_roi_box_img, target_box, "Target ROI")
    ]:
        x0, y0, x1, y1 = box

        cv2.rectangle(
            img,
            (x0, y0),
            (x1, y1),
            (0, 0, 255),
            5
        )

        cv2.putText(
            img,
            label,
            (x0, max(y0 - 20, 50)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (0, 0, 255),
            4
        )

    # ORBマッチング可視化
    result = orb_match_visualize(ref_roi, target_roi)

    # 画像保存
    cv2.imwrite(OUTPUT_REF_ROI_BOX, ref_roi_box_img)
    cv2.imwrite(OUTPUT_TARGET_ROI_BOX, target_roi_box_img)
    cv2.imwrite(OUTPUT_REF_ROI_CROP, ref_roi)
    cv2.imwrite(OUTPUT_TARGET_ROI_CROP, target_roi)
    cv2.imwrite(OUTPUT_GOOD_MATCHES, result["good_vis"])
    cv2.imwrite(OUTPUT_INLIER_MATCHES, result["inlier_vis"])

    # 結果表示
    print("========== 判定結果 ==========")
    print(f"score        : {result['score']:.3f}")
    print(f"good_matches : {result['good_matches']}")
    print(f"inliers      : {result['inliers']}")
    print(f"inlier_ratio : {result['inlier_ratio']:.3f}")

    if result["score"] >= 0.8:
        print("判定: きのこの山 OK")
    else:
        print("判定: NG")

    print()
    print("========== 出力画像 ==========")
    output_files = [
        OUTPUT_REF_ROI_BOX,
        OUTPUT_TARGET_ROI_BOX,
        OUTPUT_REF_ROI_CROP,
        OUTPUT_TARGET_ROI_CROP,
        OUTPUT_GOOD_MATCHES,
        OUTPUT_INLIER_MATCHES
    ]

    for path in output_files:
        print(path)

    print()
    print("特に確認する画像:")
    print(OUTPUT_INLIER_MATCHES)


if __name__ == "__main__":
    main()