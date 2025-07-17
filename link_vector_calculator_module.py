import numpy as np


class LinkVectorCalculator:
    """
    指定されたリンク（関節ペア）の相対位置ベクトル、速度ベクトル、角速度ベクトル、
    加速度ベクトル、角加速度ベクトル、および重心を計算・記録するクラス。

    Attributes:
        index_start (int): 始点となる関節のインデックス。
        index_end (int): 終点となる関節のインデックス。
        dt (float | None): フレーム間の時間差（delta time）。
        previous_relative_position_vector (np.ndarray | None): 直前フレームにおけるリンクの相対位置ベクトル。
        previous_velocity_vector (np.ndarray | None): 直前フレームの速度ベクトル。
        byte_vData (bool): 速度データの有無を示すフラグ（未使用の可能性あり）。
        byte_AccData (bool): 加速度データの有無を示すフラグ（未使用の可能性あり）。
        previous_position (np.ndarray | None): 直前フレームの座標データ（未使用の可能性あり）。
        previous_velocity (np.ndarray | None): 直前フレームの速度（未使用の可能性あり）。
        previous_omega (np.ndarray | None): 直前フレームの角速度。
        velocity_vector (np.ndarray | None): 現在の速度ベクトル。
        relative_position_vector (np.ndarray | None): 現在のリンクの相対位置ベクトル。
        angular_velocity (np.ndarray | None): 現在の角速度ベクトル。
        centroid (np.ndarray | None): 始点と終点の中間点（重心）。
        acceleration (np.ndarray | None): 現在の加速度ベクトル。
        angular_acceleration (np.ndarray | None): 現在の角加速度ベクトル。
    """

    def __init__(self, index_start, index_end):
        self.index_start = index_start
        self.index_end = index_end
        self.dt = None
        self.previous_relative_position_vector = None
        self.previous_velocity_vector = None
        self.byte_vData = False
        self.byte_AccData = False
        self.previous_position = None
        self.previous_velocity = None
        self.previous_omega = None
        self.velocity_vector = None
        self.relative_position_vector = None
        self.angular_velocity = None
        self.centroid = None
        self.acceleration = None
        self.angular_acceleration = None
        # print("LinkVectorCalculator object created")

    def calculate_link_vectors(self, keypoints_list, datFile_mode, i, dt):
        """
        最新のキーポイントデータを使用して、指定されたリンクの相対位置ベクトル、速度ベクトル、角速度ベクトルを計算し、
        さらに2点の重心座標を計算します。

        :param keypoints_list: キーポイントのリスト
        :return: 相対位置ベクトル、速度ベクトル、角速度ベクトル、重心座標
        """
        self.dt = dt
        if len(keypoints_list) < 2:
            # print("Data is insufficient")
            return None, None, None, None  # データが不足している場合は計算不可

        # 最新のキーポイントデータから相対位置ベクトルを計算
        # print("AAAA")
        if datFile_mode == 1:
            current_position = keypoints_list[i]
            self.relative_position_vector = (
                current_position[self.index_end] - current_position[self.index_start]
            )
            if self.previous_relative_position_vector is None:
                previous_current_position = keypoints_list[i - 1]
                self.previous_relative_position_vector = (
                    previous_current_position[self.index_end]
                    - previous_current_position[self.index_start]
                )

        else:  # リアルタイムデータの場合
            current_position = keypoints_list[-1]
            self.relative_position_vector = (
                current_position[self.index_end] - current_position[self.index_start]
            )
            if self.previous_relative_position_vector is None:
                pass
        # 重心座標を計算
        self.centroid = (
            current_position[self.index_end] + current_position[self.index_start]
        ) / 2
        # print("centroid",self.centroid)
        # 速度ベクトルの計算
        if self.previous_relative_position_vector is not None:
            self.velocity_vector = (
                self.relative_position_vector - self.previous_relative_position_vector
            ) / self.dt
        else:
            self.velocity_vector = None  # 最初の計算では速度ベクトルが未定義
        # print("velocity_vector",self.velocity_vector)
        # 速度が2つ溜まったら角速度ベクトルと加速度の計算
        if (
            self.previous_velocity_vector is not None
            and self.velocity_vector is not None
        ):
            self.angular_velocity = (
                np.cross(self.previous_velocity_vector, self.velocity_vector)
                / np.linalg.norm(self.relative_position_vector) ** 2
            )
            self.acceleration = (
                self.velocity_vector - self.previous_velocity_vector
            ) / self.dt
            # print("angular_velocity",self.angular_velocity)
        else:
            self.angular_velocity = None  # 角速度も未定義
            self.acceleration = None
        if self.previous_omega is not None:

            self.angular_acceleration = (
                self.angular_velocity - self.previous_omega
            ) / self.dt
        # print("angular_acceleration",self.angular_acceleration)

        # 状態の更新
        self.previous_relative_position_vector = self.relative_position_vector
        self.previous_velocity_vector = self.velocity_vector
        self.previous_omega = self.angular_velocity
        # print("relative_position_vector",self.relative_position_vector)
        return (
            self.relative_position_vector,
            self.velocity_vector,
            self.angular_velocity,
            self.centroid,
            current_position[self.index_start],
            self.acceleration,
            self.angular_acceleration,
        )
