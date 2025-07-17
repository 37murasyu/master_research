import numpy as np


class BodyPartDataStorage:
    """
    各身体部位の運動データ（位置、速度、角速度、加速度、トルクなど）を時系列で記録・管理するためのクラス。

    Attributes
    ----------
    storage : dict
        部位名をキーとし、各部位のデータをリストとして格納する辞書。
    """

    def __init__(self):
        self.storage = {}  # 部位ごとのデータを格納する辞書
        self.torques = {
            "wrist_R": [],
            "wrist_L": [],
            "elbow_R": [],
            "elbow_L": [],
            "core": [],
            "hip": [],
            "shoulder_R": [],
            "shoulder_L": [],
        }
        """
        BodyPartDataStorage のインスタンスを初期化する。
        各部位のデータは `storage` 属性の辞書に格納される。
        """

    def add_data(
        self,
        part_name,
        relative_position_vector,
        velocity_vector,
        omega,
        centroid,
        p1,
        dot_omega,
        dot_dot_pg,
    ):
        """
        各部位のデータを辞書に追加するメソッド。

        :param part_name: 部位名 (文字列)
        :param relative_position_vector: 相対位置ベクトル
        :param velocity_vector: 速度ベクトル
        :param omega1: 角速度ベクトル
        :param centroid: 重心座標
        :param p1: 開始点座標
        :param dot_omega: 角加速度ベクトル
        :param dot_dot_pg: 加速度ベクトル (位置ベクトルの2階微分)
        """
        # 部位名をキーとして、関連データをリストで格納
        if part_name not in self.storage:
            self.storage[part_name] = []
        if dot_omega is None:
            dot_omega = np.zeros(3)
        if dot_dot_pg is None:
            dot_dot_pg = np.zeros(3)
        self.storage[part_name].append(
            {
                "part_name": str(part_name),
                "relative_position_vector": relative_position_vector,
                "velocity_vector": velocity_vector,
                "omega": omega,
                "centroid": centroid,
                "p1": p1,
                "dot_omega": dot_omega,
                "dot_dot_pg": dot_dot_pg,
                "torque": [],  # トルクデータ用の空リストを初期化
            }
        )

    def add_torque(self, part_name: str, torque: np.ndarray):
        if part_name not in self.torques:
            self.torques[part_name] = []
        self.torques[part_name].append(torque)

    def get_data(self, part_name):
        """
        指定された部位のデータを取得するメソッド。

        :param part_name: 部位名 (文字列)
        :return: 部位のデータリスト、部位が存在しない場合は None を返す
        """
        return self.storage.get(part_name, None)

    def get_torques(self, part_name: str) -> list[np.ndarray]:
        return self.torques.get(part_name, [])
