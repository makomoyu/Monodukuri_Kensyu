import cv2
import numpy as np


def binarize_text_white(image):
    """
    画像を二値化する。
    文字部分を白255、背景を黒0にする。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 黒文字を白文字に反転
    binary_inv = 255 - binary

    return binary_inv


def crop_foreground(binary, padding=0):
    """
    白い文字部分の外接矩形で切り出す。
    """
    ys, xs = np.where(binary > 0)

    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("文字領域が見つかりません。二値化を確認してください。")

    x0 = max(xs.min() - padding, 0)
    y0 = max(ys.min() - padding, 0)
    x1 = min(xs.max() + 1 + padding, binary.shape[1])
    y1 = min(ys.max() + 1 + padding, binary.shape[0])

    return binary[y0:y1, x0:x1]


def main():
    template_path = r".\meiji_template.bmp"
    target_path = r".\kinoko\mushroom_0deg.JPG"
    template_img = cv2.imread(template_path)
    target_img = cv2.imread(target_path)

    if template_img is None:
        raise FileNotFoundError(f"テンプレート画像が読み込めません: {template_path}")

    if target_img is None:
        raise FileNotFoundError(f"検証画像が読み込めません: {target_path}")

    # 二値化
    template_binary = binarize_text_white(template_img)
    target_binary = binarize_text_white(target_img)

    # テンプレート側は白余白を削って文字部分だけにする
    template_crop = crop_foreground(template_binary, padding=0)

    print("元テンプレートサイズ:", template_binary.shape[::-1])
    print("切り出し後テンプレートサイズ:", template_crop.shape[::-1])
    print("検証画像サイズ:", target_binary.shape[::-1])

    # =========================
    # この2つの画像用のリサイズ値
    # =========================
    scale_x = 1.62
    scale_y = 1.60

    new_w = int(round(template_crop.shape[1] * scale_x))
    new_h = int(round(template_crop.shape[0] * scale_y))

    resized_template = cv2.resize(
        template_crop,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    print("リサイズ後テンプレートサイズ:", resized_template.shape[::-1])

    # テンプレートマッチング
    result = cv2.matchTemplate(
        target_binary,
        resized_template,
        cv2.TM_CCOEFF_NORMED
    )

    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    print("相関値:", max_val)
    print("検出位置:", max_loc)

    if max_val >= 0.8:
        print("OK: 相関値が0.8以上です")
    else:
        print("NG: 相関値が0.8未満です")

    # 結果画像を保存
    output = target_img.copy()

    x, y = max_loc
    h, w = resized_template.shape

    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (0, 0, 255),
        2
    )

    cv2.putText(
        output,
        f"score={max_val:.3f}",
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.imwrite("match_result.png", output)
    cv2.imwrite("template_binary.png", template_binary)
    cv2.imwrite("template_crop.png", template_crop)
    cv2.imwrite("resized_template.png", resized_template)
    cv2.imwrite("target_binary.png", target_binary)

    print("match_result.png を保存しました")


if __name__ == "__main__":
    main()