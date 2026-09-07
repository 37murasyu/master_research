import numpy as np
import os


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
        com_fraction (float): 重心比。近位端(end)から遠位端(start)へ測った比。既定 0.5。
        centroid (np.ndarray | None): 重心。``p_end + com_fraction * (p_start - p_end)``。
        centroid_velocity (np.ndarray | None): 重心の速度。
        previous_centroid (np.ndarray | None): 直前フレームの重心。
        previous_centroid_velocity (np.ndarray | None): 直前フレームの重心速度。
        acceleration (np.ndarray | None): **重心の**加速度ベクトル。
            並進の慣性力 F = m(a - g) に渡す量。リンクベクトル r の 2 階微分ではない。
        angular_acceleration (np.ndarray | None): 現在の角加速度ベクトル。
    """

    def __init__(self, index_start, index_end, com_fraction=0.5):
        self.index_start = index_start
        self.index_end = index_end
        # 重心比。近位端（end 側）から遠位端（start 側）へ測った比。
        # 0.5 なら両端の中点で、2026-09-08 以前の挙動と一致する。
        # 実測の比は上腕 0.436・前腕 0.430 で、中点だと重力モーメント腕が
        # それぞれ +14.7%・+16.3% 過大になる（再検算 R-4）。
        self.com_fraction = float(com_fraction)
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
        # 重心の履歴。dot_dot_pg（並進の慣性力 F = m(a - g) に入る加速度）は
        # 重心の 2 階微分でなければならない。かつてリンクベクトル r の 2 階微分を
        # 渡しており、始点が固定なら 2 倍、始点が動けば別のベクトルになっていた。
        self.previous_centroid = None
        self.previous_centroid_velocity = None
        self.centroid_velocity = None
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
            return None, None, None, None

        # 現在位置の選択（datFile_mode: 1=ファイル、その他=リアルタイム）
        current_position = keypoints_list[i] if datFile_mode == 1 else keypoints_list[-1]
        self.relative_position_vector = (
            current_position[self.index_end] - current_position[self.index_start]
        )
        if self.previous_relative_position_vector is None and datFile_mode == 1:
            prev = keypoints_list[i - 1]
            self.previous_relative_position_vector = (
                prev[self.index_end] - prev[self.index_start]
            )
        # 重心座標を計算。近位端（end）から com_fraction だけ遠位端（start）寄り。
        self.centroid = current_position[self.index_end] + self.com_fraction * (
            current_position[self.index_start] - current_position[self.index_end]
        )
        if self.previous_centroid is None and datFile_mode == 1:
            prev = keypoints_list[i - 1]
            self.previous_centroid = prev[self.index_end] + self.com_fraction * (
                prev[self.index_start] - prev[self.index_end]
            )
        # print("centroid",self.centroid)
        # 速度ベクトルの計算
        self.velocity_vector = (
            (self.relative_position_vector - self.previous_relative_position_vector) / self.dt
            if self.previous_relative_position_vector is not None
            else None
        )
        # print("velocity_vector",self.velocity_vector)
        # 角速度は標準形 omega = (r x r') / |r|^2。
        # 剛体回転 r' = omega x r を代入すると r x r' = omega_perp |r|^2 になる
        # （リンク軸まわりの回転は 2 点からは取れないので常に 0 になる。これは原理的な限界）。
        # かつて第 1 引数に前フレームの速度を入れており、返していたのは
        # dt |omega_perp|^2 omega_perp という次元 1/s^2 の別物だった。信号は
        # omega^2 dt 倍に潰れ、逆に位置ノイズは (sigma/dt)^2/|r|^2 で増幅されていた。
        # r' が 1 つあれば求まるので、速度が溜まった時点で計算できる。
        if self.velocity_vector is not None:
            # 安全ガード: リンク長が極小/NaN の場合はゼロ割を避ける
            denom = np.linalg.norm(self.relative_position_vector)
            if not np.isfinite(denom) or denom < 1e-9:
                # デバッグ出力は既定OFF（必要時は DEBUG_LOGS=1 または LVC_DEBUG=1）
                if os.getenv('LVC_DEBUG', '0') not in ('0','false','False') or os.getenv('DEBUG_LOGS', '0') not in ('0','false','False'):
                    try:
                        print(f"[LVC] tiny|bad denom at idx({self.index_start}->{self.index_end}): |r|={denom}")
                    except Exception:
                        pass
                self.angular_velocity = np.array([0.0, 0.0, 0.0])
            else:
                cross_val = np.cross(self.relative_position_vector, self.velocity_vector)
                if not np.all(np.isfinite(cross_val)):
                    if os.getenv('LVC_DEBUG', '0') not in ('0','false','False') or os.getenv('DEBUG_LOGS', '0') not in ('0','false','False'):
                        try:
                            print(f"[LVC] non-finite cross at idx({self.index_start}->{self.index_end})")
                        except Exception:
                            pass
                    self.angular_velocity = np.array([0.0, 0.0, 0.0])
                else:
                    self.angular_velocity = cross_val / (denom ** 2)
        else:
            self.angular_velocity = None

        # 重心の速度と加速度。並進の慣性力 F = m(a - g) に渡すのはこちらであって、
        # リンクベクトル r の 2 階微分ではない。r'' = p_end'' - p_start'' なので
        # 始点が固定なら重心加速度のちょうど 2 倍、始点が動けば別のベクトルになる。
        self.centroid_velocity = (
            (self.centroid - self.previous_centroid) / self.dt
            if self.previous_centroid is not None
            else None
        )
        self.acceleration = (
            (self.centroid_velocity - self.previous_centroid_velocity) / self.dt
            if (self.previous_centroid_velocity is not None and self.centroid_velocity is not None)
            else None
        )
        if self.previous_omega is not None:
            self.angular_acceleration = (
                self.angular_velocity - self.previous_omega
            ) / self.dt
        # print("angular_acceleration",self.angular_acceleration)

        # 状態の更新
        self.previous_relative_position_vector = self.relative_position_vector
        self.previous_velocity_vector = self.velocity_vector
        self.previous_omega = self.angular_velocity
        self.previous_centroid = self.centroid
        self.previous_centroid_velocity = self.centroid_velocity
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
