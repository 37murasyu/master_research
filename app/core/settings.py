"""146 箇所に散った ``os.getenv`` を、型付きの設定として外から束ねる。

``master_research_code.py`` は設定をすべて環境変数で受け取る作りになっている
（``os.getenv`` が 150 箇所、ユニーク 135 個）。この層はそれを

1. 型付きスキーマとして表現し
2. ユーザ領域の JSON に永続化し
3. **環境変数の辞書として子プロセスに渡す**

という形で扱う。3 の経路を取るので、**既存コードは 1 行も変更しなくてよい**。

スキーマの雛形は ``tools/extract_env_schema.py`` がソースから機械生成し、
``settings_schema.json`` に置いてある。UI に出す項目・説明文・アプリ側の
既定値の上書きは、このファイルの ``CURATED`` で行う。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from app.core.platform_compat import user_config_dir, user_output_dir

__all__ = ["Setting", "Settings", "SCHEMA", "APP_NAME", "OUTPUT_DIR_ENV", "measurement_output_dir"]

APP_NAME = "WheelchairTorque"

# 計測の CSV の置き場を GUI からワーカーへ伝える環境変数（config.save_dir が読む）。設定画面の項目ではない
OUTPUT_DIR_ENV = "OUTPUT_DIR"


def measurement_output_dir() -> Path:
    """計測（master_research_code.py）の CSV の置き場。GUI の表示・解析の既定の入力フォルダ・ワーカーへの
    受け渡しは、すべてここから取る（かつて表示と実際が食い違っていた。tests/test_output_dir.py）。

    凍結アプリのワークスペース（同じ ``user_output_dir``）の ``output_data`` と同じ場所。
    """
    return user_output_dir(APP_NAME) / "output_data"

_SCHEMA_FILE = Path(__file__).with_name("settings_schema.json")

# 既存コードが bool を読むときの慣用句。``os.getenv(X, '1') in ('1','true','True')``
_TRUE_LITERALS = ("1", "true", "True")


@dataclass(frozen=True)
class Setting:
    """設定 1 項目の定義。"""

    name: str
    type: str  # "bool" | "int" | "float" | "str"
    code_default: str | None
    group: str
    description: str = ""
    ui_visible: bool = False
    # アプリとしての既定値。コード側の既定が不適切な場合にここで上書きする。
    app_default: str | None = None

    @property
    def effective_default(self) -> str | None:
        """実際に使う既定値。アプリ側の指定があればそちらが勝つ。"""
        return self.app_default if self.app_default is not None else self.code_default


# ---------------------------------------------------------------------------
# 手で面倒を見る項目
#
# 大半の設定は機械生成のままで足りる。ここに書くのは
#   (a) コード側の既定値が壊れていてアプリ側で変える必要があるもの
#   (b) 利用者が UI から触る必要があるもの
# のいずれか。
# ---------------------------------------------------------------------------
CURATED: dict[str, dict[str, Any]] = {
    # --- (a) 既定値が壊れている 4 つ -------------------------------------
    # 素の `python master_research_code.py` はこれらが有効なため、
    # 警告も出さずに無意味な出力を作る。アプリでは既定で無効にする。
    "DEMO_MONO_GAUGE_ON": {
        "app_default": "0",
        "ui_visible": True,
        "description": (
            "デモ用の単眼ゲージ表示。有効だと逆動力学の計算が丸ごと止まり、"
            "トルクCSVが全ゼロで出力される（警告は出ない）。通常は無効。"
        ),
    },
    "DEMO_MONO_CAM0_ONLY": {
        "app_default": "0",
        "ui_visible": True,
        "description": (
            "カメラ0の映像を1にも複製するデモ用モード。有効だと同一画像を"
            "異なる投影行列で三角測量することになり、3D再構成が無意味になる。通常は無効。"
        ),
    },
    "RT_POSE_FIXED_HZ_ON": {
        "app_default": "0",
        "ui_visible": True,
        "description": (
            "姿勢推定を固定レート（既定4Hz）に間引く。有効だと実処理が約3.75Hzに"
            "落ちる一方、エネルギー計算は30Hz前提のままなのでカットオフ周波数が8倍ずれる。"
        ),
    },
    "E_LPF_NATIVE_ON": {
        "app_default": "0",
        "ui_visible": True,
        "description": (
            "ローパスフィルタをネイティブ実装（1次指数フィルタ）に差し替える。"
            "有効だと Butterworth filtfilt ではなくなり、既発表の数値と比較できなくなる。"
        ),
    },
    # --- (b) 利用者が触る項目 --------------------------------------------
    # 被験者番号。これを渡さないと master_research_code.py:905 が input() で
    # 標準入力を待つ。GUI から起動すると端末が無いのでハングする
    # （ランナー側で stdin を閉じる保険も入れてあるが、値は明示的に渡すべき）。
    "SUBJECT_ID": {
        "type": "str",
        "ui_visible": True,
        "group": "被験者",
        "description": (
            "被験者番号（例: 00）。Mac＋Pixel は m_max_all_merged.csv の番号（整数）で 1RM を引き、"
            "ゲージの目標帯（論文の W_0.70〜W_0.85）を決める。USB カメラ 2 台は m_max_part_<番号>.json を読む。"
        ),
    },
    # --- 混成（Mac＋Pixel）だけが読む項目 ----------------------------------
    # スキーマの生成元は master_research_code.py と config.py しか見ないので、ここに足さないと GUI から渡らない。
    # 名前を HYBRID_ で始めるのは、GUI が全件渡す USB 向けの既定（POSE_ROI_ON=1・EKF_Q_ACC=1e-3 など）を
    # 混成に漏らさないため（tests/test_hybrid_settings.py）。
    "HYBRID_EKF_PROFILE": {
        "type": "str",
        "code_default": "",
        "ui_visible": True,
        "group": "カルマンフィルタ",
        "description": (
            "Mac＋Pixel の EKF の較正プロファイル（ファイルかフォルダ）。空なら同梱の既定値。"
            "解析ページで計測フォルダの kpts3d_raw_*.csv から作れる（処理の間隔 1/30 秒のものだけが選ばれる）。"
        ),
    },
    "HYBRID_GRAVITY_BOARD": {
        "type": "bool",
        "code_default": "1",
        "ui_visible": True,
        "group": "混成ステレオ",
        "description": "Mac＋Pixel の校正の最後に、盤を立てて重力の向きを記録する（Enter で省略できる）。",
    },
    "HYBRID_GRAVITY_BOARD_TIMEOUT_S": {
        "type": "float",
        "code_default": "30",
        "group": "混成ステレオ",
        "description": "盤を立てる段階を打ち切るまでの秒数。",
    },
    "HYBRID_POSE_MODEL": {
        "type": "str",
        "code_default": "",
        "group": "混成ステレオ",
        "description": "Mac 側の姿勢推定のモデル（.task）。空なら同梱の lite（Pixel と同じ）。",
    },
    "HYBRID_POSE_MIN_DET": {
        "type": "float",
        "code_default": "0.5",
        "group": "混成ステレオ",
        "description": "Mac 側の姿勢推定の検出の閾値。",
    },
    "HYBRID_POSE_MIN_PRESENCE": {
        "type": "float",
        "code_default": "0.5",
        "group": "混成ステレオ",
        "description": "Mac 側の姿勢推定の存在の閾値。",
    },
    "HYBRID_POSE_MIN_TRACK": {
        "type": "float",
        "code_default": "0.5",
        "group": "混成ステレオ",
        "description": "Mac 側の姿勢推定の追跡の閾値。",
    },
    "HYBRID_POSE_INPUT_SCALE": {
        "type": "float",
        "code_default": "1.0",
        "group": "混成ステレオ",
        "description": "Mac 側の姿勢推定に渡す画像の縮小率（0.25〜1）。",
    },
    "HYBRID_POSE_ROI": {
        "type": "bool",
        "code_default": "0",
        "group": "混成ステレオ",
        "description": "Mac 側の姿勢推定で人の周りだけを切り出して推定する（既定は無効。有効にすると追跡を使わない IMAGE モードになり、Pixel 側と推定の条件が変わるので試すときは比べて確かめる）。",
    },
    "HYBRID_DYN_GATE": {
        "type": "bool",
        "code_default": "1",
        "group": "混成ステレオ",
        "description": (
            "押し上げの間だけ仕事とゲージを積む（座っている間の雑音を積まない）。"
            "トルクは関所によらず記録する。"
        ),
    },
    "ONE_RM_CSV": {
        "type": "str",
        "group": "被験者",
        "description": "1RM の表のパス。空なら作業フォルダの m_max_all_merged.csv。",
    },
    # config.py 側は int(os.environ.get(...)) で読むが、実質は 0/1 の真偽値。
    # UI ではチェックボックスにする。"1"/"0" を渡せば int() は問題なく解釈する。
    "HEADLESS": {
        "type": "bool",
        "ui_visible": True,
        "group": "表示",
        "description": "ウィンドウを一切開かずに実行する。計測を裏で走らせたいときに使う。",
    },
    "USE_SAMPLE_VIDEOS": {
        "type": "bool",
        "ui_visible": True,
        "group": "入力ソース",
        "description": "カメラの代わりに収録済み動画を入力にする。動作確認や再解析に使う。",
    },
    "AUTO_FALLBACK_TO_FILES": {
        "type": "bool",
        "ui_visible": True,
        "group": "入力ソース",
        "description": "カメラを開けなかったとき、自動で動画ファイルに切り替える。",
    },
    "CAM0": {
        "ui_visible": True,
        "group": "カメラ",
        "description": "カメラ0の指定。数字ならデバイス番号、それ以外は名前やパスとして扱う。",
    },
    "CAM1": {
        "ui_visible": True,
        "group": "カメラ",
        "description": "カメラ1の指定。数字ならデバイス番号、それ以外は名前やパスとして扱う。",
    },
    "IO_DEBUG": {
        "type": "bool",
        "ui_visible": True,
        "group": "診断",
        "description": "入出力まわりの詳細ログを出す。カメラが開かないときの切り分けに使う。",
    },
    # 較正プロファイル（設計メモ 実装 5、S9）。指定するとプロファイルが EKF_Q_ACC / EKF_R より
    # 優先する。EKF_Q_ACC / EKF_R は UI に出さない（widgets の小数 4 桁で推定値 r≈2.6e-5 が 0 に丸まる）
    "EKF_PROFILE": {
        "type": "str",
        "ui_visible": True,
        "group": "カルマンフィルタ",
        "description": (
            "EKF の較正プロファイル（ファイルかフォルダ）。フォルダなら ekf_profile_*.json のうち"
            "処理の間隔が合うものを使う。空なら従来の固定値。プロファイルは解析ページで生 CSV から作れる。"
        ),
    },
    # ゲージ画面の項目。スキーマの生成元（master_research_code.py・config.py）は
    # ゲージ表示を知らないので、GAUGE_SHOW_JOULES はここだけで定義する（_load_schema が
    # CURATED 専用の Setting を組み立てる）。操作口はゲージ画面のスイッチだけにするため、
    # 設定フォームには出さない（ui_visible=False）。
    "GAUGE_SHOW_JOULES": {
        "type": "bool",
        "code_default": "1",
        "group": "表示",
        "description": "ゲージに J（仕事）の数値を表示する。",
    },
    "BODY_MASS_KG": {
        "ui_visible": True,
        "group": "被験者",
        "description": "体重 [kg]。",
    },
    # --- 記録の再生（role hybrid_replay）で何を流すか -----------------------
    # 次回も同じフォルダ・範囲・速さで流せるよう設定に覚えておく。子へは環境変数ではなく引数で渡す
    # （計測画面が app.shell.page_measure.replay_arguments で組み立てる。子は環境変数から読まない）。
    # フォルダは計測画面の専用の欄（入力「記録の再生」）で選ぶので、設定フォームには出さない。
    "HYBRID_REPLAY": {
        "type": "str",
        "code_default": "",
        "group": "記録の再生",
        "description": "流し直す Mac＋Pixel の計測フォルダ（meta.json のあるもの）。",
    },
    "HYBRID_REPLAY_FROM": {
        "type": "float",
        "code_default": "0",
        "ui_visible": True,
        "group": "記録の再生",
        "description": "記録の何秒目から流すか。",
    },
    # 数の欄では「終わりまで」を表せないので文字の欄にする（空なら終わりまで。子は数として読む）
    "HYBRID_REPLAY_TO": {
        "type": "str",
        "code_default": "",
        "ui_visible": True,
        "group": "記録の再生",
        "description": "記録の何秒目まで流すか。空なら終わりまで。",
    },
    "HYBRID_REPLAY_SPEED": {
        "type": "float",
        "code_default": "1",
        "ui_visible": True,
        "group": "記録の再生",
        "description": "流す速さ。1 で実時間、0 で待たない。",
    },
    # ゲージの書体の組（app.gauge.fonts.PRESETS）。Mac に入っているフォントワークスの書体を使う。
    # 入っていない書体はヒラギノ → 同梱の IPAex ゴシックへ落ちる。手元の見え方は
    # `python -m app.gauge.fonts` で確かめられる。
    "GAUGE_FONT_PRESET": {
        "type": "str",
        "code_default": "rodin",
        "ui_visible": True,
        "group": "表示",
        "description": (
            "ゲージの書体の組。rodin（見出しロダン＋UD角ゴ_ラージ）／tsukushi（見出し筑紫A見出ミン）／"
            "kaimin（見出し解ミン 宙）／system（従来のヒラギノ角ゴ）。"
        ),
    },
}


def _load_schema() -> dict[str, Setting]:
    payload = json.loads(_SCHEMA_FILE.read_text(encoding="utf-8"))
    schema: dict[str, Setting] = {}

    for name, raw in payload["settings"].items():
        schema[name] = Setting(
            name=name,
            type=raw["type"],
            code_default=raw.get("default"),
            group=raw.get("group", "その他"),
        )

    for name, overrides in CURATED.items():
        base = schema.get(name)
        if base is None:
            # 生成元に無い設定も UI に出せるようにしておく
            base = Setting(
                name=name,
                type=overrides.get("type", "str"),
                code_default=overrides.get("code_default"),
                group=overrides.get("group", "その他"),
            )
        # 型も上書きできる。機械推定では int になるが実質は真偽値、という項目が
        # あるため（config.py の int(os.environ.get("HEADLESS", "0")) など）。
        schema[name] = replace(base, **overrides)

    return schema


SCHEMA: dict[str, Setting] = _load_schema()


def _to_str(setting: Setting, value: Any) -> str:
    """Python の値を、既存コードが読める文字列に変換する。"""
    if setting.type == "bool":
        if not isinstance(value, bool):
            raise TypeError(f"{setting.name} は bool。受け取った値: {value!r}")
        return "1" if value else "0"
    if setting.type == "int":
        return str(int(value))
    if setting.type == "float":
        return repr(float(value))
    return str(value)


def _from_str(setting: Setting, raw: str) -> Any:
    if setting.type == "bool":
        return raw in _TRUE_LITERALS
    if setting.type == "int":
        return int(raw)
    if setting.type == "float":
        return float(raw)
    return raw


class Settings:
    """設定値の集合。既定値との差分だけを保持する。"""

    def __init__(self, values: dict[str, str] | None = None):
        # 既定と異なる項目のみを持つ。既定が変わったとき自動で追随できるようにするため。
        self._overrides: dict[str, str] = {}
        for name, raw in (values or {}).items():
            if name in SCHEMA:  # 消えた設定は黙って捨てる（古い保存ファイル対策）
                self._overrides[name] = raw

    # -- 参照・更新 --------------------------------------------------------
    def _setting(self, name: str) -> Setting:
        try:
            return SCHEMA[name]
        except KeyError:
            raise KeyError(
                f"未知の設定です: {name}\n"
                f"  スキーマは {_SCHEMA_FILE.name} にあります。"
                f"（tools/extract_env_schema.py で再生成できます）"
            ) from None

    def get(self, name: str) -> Any:
        setting = self._setting(name)
        raw = self._overrides.get(name, setting.effective_default)
        if raw is None:
            return None
        return _from_str(setting, raw)

    def set(self, name: str, value: Any) -> None:
        setting = self._setting(name)
        raw = _to_str(setting, value)
        if raw == setting.effective_default:
            self._overrides.pop(name, None)  # 既定に戻ったら差分から外す
        else:
            self._overrides[name] = raw

    @property
    def overrides(self) -> dict[str, str]:
        """既定と異なる項目だけ。"""
        return dict(self._overrides)

    # -- 子プロセスへの受け渡し -------------------------------------------
    def as_env(self) -> dict[str, str]:
        """環境変数の辞書。**全設定を明示的に含める**。

        差分だけ渡すと、渡さなかった項目は子プロセス側の既定値が効いてしまう。
        その既定値こそが壊れているものを含むので、全件を明示して
        呼び出し元のシェル環境からも隔離する。
        """
        env: dict[str, str] = {}
        for name, setting in SCHEMA.items():
            raw = self._overrides.get(name, setting.effective_default)
            if raw is not None:
                env[name] = raw
        return env

    # -- 永続化 ------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_note": "既定値と異なる項目のみ保存している。既定はアプリ側で管理。",
            "values": self._overrides,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "Settings":
        path = Path(path)
        if not path.is_file():
            return cls()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # 壊れた設定ファイルで起動できなくなるより、既定で立ち上がる方がよい。
            return cls()
        return cls(payload.get("values", {}))

    @classmethod
    def default_path(cls) -> Path:
        return user_config_dir(APP_NAME) / "settings.json"
