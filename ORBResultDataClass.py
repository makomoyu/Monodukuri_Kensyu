from dataclasses import dataclass, field
from typing import List
import cv2

@dataclass
class ResultData:
    """
    ORBマッチングの結果をまとめたデータクラス
    """

    Score: float = 0.0 # 最終的な一致スコア（0.0〜1.0）
    GoodMatches: int = 0 # 似ている特徴点のマッチ数
    Inliers: int = 0 # 幾何的に正しいと判断されたマッチ数
    InlierRatio: float = 0.0 # Inliers / GoodMatches（マッチの信頼度）
    KeypointsRef: List[cv2.KeyPoint] = field(default_factory=list) # 正解画像の特徴点
    KeypointsTarget: List[cv2.KeyPoint] = field(default_factory=list) # 検査対象画像の特徴点
    GoodMatchList: List[cv2.DMatch] = field(default_factory=list) # GoodMatchesの内訳（マッチ一覧）
    InlierMatchList: List[cv2.DMatch] = field(default_factory=list) # 幾何的に正しいマッチの一覧（最も重要）