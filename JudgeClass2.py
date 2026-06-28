import cv2
import numpy as np
import time
from SaveImage import SaveImage
from RotationClass import ImageRotator
from ORBResultDataClass import ResultData

SAVE_FOLDER_PATH = r".\check_image"

def convert_to_gray_image(color_or_gray_image):
    """カラー画像ならグレースケールへ変換する"""
    if color_or_gray_image is None:
        raise ValueError("画像が None です")
    if len(color_or_gray_image.shape) == 3:
        return cv2.cvtColor(color_or_gray_image, cv2.COLOR_BGR2GRAY)
    return color_or_gray_image.copy()

def apply_clahe_to_gray_image(gray_image):
    """グレースケール画像にCLAHEを適用する"""
    clahe_filter = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8, 8))
    return clahe_filter.apply(gray_image)

def preprocess_image_for_orb(input_image):
    """ORB特徴量抽出用に画像を前処理する"""
    gray_image = convert_to_gray_image(input_image)
    enhanced_gray_image = apply_clahe_to_gray_image(gray_image)

    SaveImage.save(SAVE_FOLDER_PATH, "gray.bmp", enhanced_gray_image)

    blurred_gray_image = cv2.GaussianBlur(enhanced_gray_image,(3, 3),0)
    return blurred_gray_image

def create_orb_detector():
    """ORB検出器を生成する"""
    return cv2.ORB_create(
        nfeatures=3000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        patchSize=31
    )

def detect_orb_keypoints_and_descriptors(input_image):
    """ORBのキーポイントと特徴量を取得する"""
    preprocessed_image = preprocess_image_for_orb(input_image)
    orb_detector = create_orb_detector()
    keypoints, descriptors = orb_detector.detectAndCompute(preprocessed_image,None)
    return keypoints, descriptors

def match_descriptors_by_knn(reference_descriptors, target_descriptors):
    """KNNマッチングで特徴量を対応付ける"""
    brute_force_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    return brute_force_matcher.knnMatch(reference_descriptors,target_descriptors,k=2)

def filter_good_matches_by_lowe_ratio(knn_matches, ratio_threshold=0.75):
    """Loweのratio testで良いマッチのみ抽出する"""
    good_matches = []

    for match_pair in knn_matches:
        if len(match_pair) != 2:
            continue

        best_match, second_best_match = match_pair
        if best_match.distance < ratio_threshold * second_best_match.distance:
            good_matches.append(best_match)
    return good_matches

def create_empty_result(reference_keypoints, target_keypoints):
    """特徴量が取得できなかった場合の空結果を作る"""
    return ResultData(
        KeypointsRef=reference_keypoints,
        KeypointsTarget=target_keypoints
    )

def create_low_match_result(reference_keypoints, target_keypoints, good_matches):
    """マッチ数が少ない場合の結果を作る"""
    return ResultData(
        GoodMatches=len(good_matches),
        KeypointsRef=reference_keypoints,
        KeypointsTarget=target_keypoints,
        GoodMatchList=good_matches
    )

def create_homography_from_matches(reference_keypoints, target_keypoints, good_matches):
    """良いマッチからホモグラフィを算出する"""
    reference_points = np.float32([reference_keypoints[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    target_points = np.float32([target_keypoints[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    return cv2.findHomography(
        reference_points,
        target_points,
        cv2.RANSAC,
        5.0
    )

def extract_inlier_matches(good_matches, homography_mask):
    """RANSACのmaskからインライアのマッチのみ抽出する"""
    mask_boolean_array = homography_mask.ravel().astype(bool)

    return [
        match
        for match, is_inlier in zip(good_matches, mask_boolean_array)
        if is_inlier
    ]

def calculate_orb_similarity_score(good_match_count, inlier_count):
    """インライア数と比率からORB類似度スコアを算出する"""
    inlier_ratio = inlier_count / good_match_count if good_match_count > 0 else 0.0

    inlier_score = min(inlier_count / 120.0, 1.0)
    ratio_score = min(inlier_ratio / 0.75, 1.0)

    similarity_score = 0.70 * inlier_score + 0.30 * ratio_score

    if inlier_count < 35:
        similarity_score *= inlier_count / 35.0

    if good_match_count < 50:
        similarity_score *= good_match_count / 50.0

    return float(max(0.0, min(similarity_score, 1.0)))

def create_success_result(reference_keypoints, target_keypoints, good_matches, inlier_matches):
    """ORB判定が成立した場合のResultDataを作成する"""
    good_match_count = len(good_matches)
    inlier_count = len(inlier_matches)
    inlier_ratio = inlier_count / good_match_count if good_match_count > 0 else 0.0

    similarity_score = calculate_orb_similarity_score(
        good_match_count,
        inlier_count
    )

    return ResultData(
        Score=similarity_score,
        GoodMatches=good_match_count,
        Inliers=inlier_count,
        InlierRatio=inlier_ratio,
        KeypointsRef=reference_keypoints,
        KeypointsTarget=target_keypoints,
        GoodMatchList=good_matches,
        InlierMatchList=inlier_matches
    )

def calculate_orb_result(reference_image, target_image):
    """参照画像と対象画像のORB類似度結果を算出する"""
    reference_keypoints, reference_descriptors = detect_orb_keypoints_and_descriptors(reference_image)
    target_keypoints, target_descriptors = detect_orb_keypoints_and_descriptors(target_image)

    if reference_descriptors is None or target_descriptors is None:
        return create_empty_result(reference_keypoints, target_keypoints)

    knn_matches = match_descriptors_by_knn(reference_descriptors,target_descriptors)
    good_matches = filter_good_matches_by_lowe_ratio(knn_matches)

    if len(good_matches) < 8:
        return create_low_match_result(reference_keypoints, target_keypoints, good_matches)

    homography_matrix, homography_mask = create_homography_from_matches(
        reference_keypoints,
        target_keypoints,
        good_matches
    )

    if homography_matrix is None or homography_mask is None:
        return create_low_match_result(reference_keypoints, target_keypoints, good_matches)

    inlier_matches = extract_inlier_matches(good_matches, homography_mask)

    return create_success_result(
        reference_keypoints,
        target_keypoints,
        good_matches,
        inlier_matches
    )

def normalize_reference_images(reference_images):
    """参照画像をリスト形式に統一する"""
    if isinstance(reference_images, list):
        return reference_images

    return [reference_images]

def find_best_orb_result(target_image, reference_images):
    """複数の参照画像から最もスコアが高い結果を取得する"""
    best_result = ResultData()
    best_reference_image = None
    best_reference_index = None

    for reference_index, reference_image in enumerate(reference_images):
        if reference_image is None:
            continue

        current_result = calculate_orb_result(reference_image, target_image)

        if current_result.Score > best_result.Score:
            best_result = current_result
            best_reference_image = reference_image
            best_reference_index = reference_index

    return best_result, best_reference_image, best_reference_index

def create_judgement_dictionary(is_ok, best_result):
    """判定結果の辞書を作成する"""
    return {
        "判定": "OK" if is_ok else "NG",
        "スコア": round(best_result.Score, 2)
    }

def select_draw_matches(best_result, max_draw_count=100):
    """描画に使うインライアマッチを距離順で取得する"""
    sorted_inlier_matches = sorted(
        best_result.InlierMatchList,
        key=lambda match: match.distance
    )

    return sorted_inlier_matches[:max_draw_count]

def draw_orb_matches(reference_image, target_image, best_result):
    """ORBのマッチング結果画像を描画する"""
    draw_matches = select_draw_matches(best_result)

    return cv2.drawMatches(
        reference_image,
        best_result.KeypointsRef,
        target_image,
        best_result.KeypointsTarget,
        draw_matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

def create_result_text(is_ok, best_result):
    """結果画像に表示するテキストを作成する"""
    label = "OK" if is_ok else "NG"

    return (
        f"{label} "
        f"score={best_result.Score:.3f} "
        f"good={best_result.GoodMatches} "
        f"inliers={best_result.Inliers} "
        f"ratio={best_result.InlierRatio:.3f}"
    )

def draw_result_text_on_image(result_image, result_text):
    """結果画像の上部に判定テキストを描画する"""
    rectangle_width = min(result_image.shape[1], 1400)
    cv2.rectangle(result_image,(0, 0),(rectangle_width, 60),(0, 0, 0),-1)
    cv2.putText(result_image,result_text,(20, 40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0, 255, 255),2)

    return result_image

def create_orb_result_image(reference_image, target_image, best_result, is_ok):
    """ORB判定結果の可視化画像を作成する"""
    result_image = draw_orb_matches(reference_image,target_image,best_result)
    result_text = create_result_text(is_ok, best_result)
    return draw_result_text_on_image(
        result_image,
        result_text
    )

def judge_image_by_orb(target_image, reference_images, threshold=0.8):
    """ORBスコアによって対象画像をOK/NG判定する"""
    if target_image is None:
        raise ValueError("target_image が None です")

    normalized_reference_images = normalize_reference_images(reference_images)
    best_result, best_reference_image, _ = find_best_orb_result(
        target_image,
        normalized_reference_images
    )

    if best_reference_image is None:
        raise ValueError("有効な reference_image がありません")

    is_ok = best_result.Score >= threshold
    result_dictionary = create_judgement_dictionary(is_ok, best_result)

    result_image = create_orb_result_image(
        best_reference_image,
        target_image,
        best_result,
        is_ok
    )

    return result_dictionary, result_image

class ORBJudge:
    """ORB画像判定クラス"""

    @staticmethod
    def result(target_color_image, reference_color_image):
        """ORBの判定結果を取得する"""
        start_time = time.perf_counter()
        result_dictionary, result_image = judge_image_by_orb(target_color_image,reference_color_image,threshold=0.8)
        SaveImage.save(SAVE_FOLDER_PATH, "result.bmp", result_image)
        end_time = time.perf_counter()
        print(f"判定処理時間：{end_time - start_time:.3f}")
        return result_dictionary

    @staticmethod
    def print_result(result_dictionary):
        """判定結果をコンソールに表示する"""
        print("=" * 20)
        for key, value in result_dictionary.items():
            print(f"{key}:{value}")
        print("=" * 20)


def load_required_images(test_image_path, template_image_path):
    """対象画像とテンプレート画像を読み込む"""
    test_image = cv2.imread(test_image_path)
    template_image = cv2.imread(template_image_path)
    if test_image is None:
        raise FileNotFoundError(f"対象画像が読み込めません: {test_image_path}")
    if template_image is None:
        raise FileNotFoundError(f"テンプレート画像が読み込めません: {template_image_path}")
    return test_image, template_image


def main():
    """メイン処理"""
    start_time = time.perf_counter()
    # 画像の読み込み
    test_image_path = r".\test_image\kinoko\a.bmp"
    template_image_path = r".\template_image\kinoko_template.bmp"
    test_image, template_image = load_required_images(test_image_path,template_image_path)

    # 結果出力
    normal_result = ORBJudge.result(test_image, template_image)
    rotated_result = ORBJudge.result(test_image, ImageRotator.rotate(template_image, 180))
    ORBJudge.print_result(normal_result)
    ORBJudge.print_result(rotated_result)

    end_time = time.perf_counter()
    print(f"合計処理時間：{end_time - start_time:.3f}")


if __name__ == "__main__":
    main()