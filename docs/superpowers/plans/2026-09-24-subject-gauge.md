# 被験者ゲージ UI 実装計画（2026-09-24）

**Goal**: 被験者が第 2 モニタで見る扇形ゲージ（混成経路）と、実験者の計測画面（設計書 §5.2）を PySide6 で実装し、`.app` で動かす。

**Spec**: `docs/superpowers/specs/2026-09-23-subject-gauge-design.md`。ただし下の「Global Constraints」が §6.3・§9 などより優先する（別セッション master-research-10 との合意）。

**作業場所**: 作業ツリー `/Users/s.murayama/projects/master_research-gauge`、ブランチ `murayama/subject-gauge`。Python は `/Users/s.murayama/projects/master_research/.venv/bin/python`（以下 `$PY`）。試験は作業ツリーの直下で `$PY -m pytest tests -q -p no:cacheprovider`。

## Global Constraints

- **ほかの作業ツリーには書き込まない**: `/Users/s.murayama/projects/master_research`（別セッションが作業中）のファイル・索引・stash には触らない。push しない。merge や rebase は、controller の指示があるときだけ行う。
- **触らないファイル**（別セッションの持ち分）: `app/gauge/tracker.py`、`app/gauge/thresholds.py`、`app/runners/network_measure.py`、`app/runners/hybrid_measure.py`、`app/hybrid/*`、`KNOWN_ISSUES.md`、`master_research_code.py`、`Gauge_display.py`。
- **試験のファイル名**: `tests/test_gauge_*.py` にする（`tests/test_protocol.py` は既に別物がある）。
- **共有の一覧に足すとき**: `app/core/settings.py` の CURATED と `tests/test_cross_platform.py` の一覧に足す項目は、どちらも**末尾に**足す。
- **行の形式 v2**（`app/gauge/protocol.py`。向こうと合意した契約）:
  - 1 行 = `@@GAUGE ` ＋ JSON ＋ `\n`。例: `{"v":2,"link":"waiting|connected","rep":N,"source":"measure|demo|replay","parts":{"elbow_L":{"now":12.3,"prev":10.1,"band":[lo,hi],"w1rm":66.0}, "elbow_R":…, "wrist_L":…, "wrist_R":…}}`
  - `now`・`prev`・`w1rm` は数か null（NaN は null で来る）。`band` は `[lo, hi]`（lo < hi）か null。`rep` は完了した回数（0 以上の整数）。`prev` は完了した回が無ければ null。
  - 読む側は、知らない鍵・知らない部位を無視する。
  - encode は小数 1 桁に丸め、区切りを詰め、ASCII で、**1 行 512 バイト未満**にする（macOS の PIPE_BUF。QProcess の MergedChannels で標準エラーと混ざらない）。
- **閾値**: 子プロセスが論文の値の W_0.70 と W_0.85 を計算して `band` で送る。GUI は `gauge_layout.json` の固定値を使わない。
- **状態の判定**: `now < lo` は不足（状態の文字なし）、`lo ≤ now < hi` は「✓ 目標帯」、`now ≥ hi` は「✕ 過負荷」。
- **弧の割合**: `f = clamp(now / (1.25·hi), 0, 1)`。右端＝hi×1.25、超えた分は右端で止める。
- **描かないもの**: band が null の部位は、溝と部位名だけ（J がオンなら中央の値の数字も）を描く。now が null の部位は値なし（溝と帯だけ）。
- **source**: "replay" のときは見出しに「▶ 再生」の状態（R17-05 のピル）を出す。"demo" は表示に使わない。
- **ゲージの設計座標は 800×450**。正本は画面案の生成スクリプト `/private/tmp/claude-502/-Users-s-murayama-projects-master-research/6b37504a-d68f-4a57-b61b-c9c0b1d724f2/scratchpad/mk_subject.py` の `gauge()`・`FIG`・`ICON`・`header` 系。画像は同じフォルダの `1_被験者ゲージ_計測中.png`・`2_被験者ゲージ_状態別.png`・`3_実験者_計測画面.png`。
- **色**: 地 `#0b1633`、溝 `#1e293b`、帯 `#1d4ed8`、帯の中 `#60a5fa`、値の弧 `#ffffff`、過負荷 `#ef4444`、文字 `#ffffff`、補足 `#94a3b8`、見出し帯 `#2563eb`、見出しの補足 `#dbeafe`、琥珀 `#fbbf24`、椅子 `#93c5fd`、プレート `#dc2626`・`#991b1b`、スリーブ `#e5e7eb`。**緑は使わない**。色だけで状態を示さない（記号と文字を添える）。
- **人物**: 頭から足先まで一続きに描く（胴・膝・すねをつなぐ）。描く順は「椅子のパッドとプレートとスリーブ → 脚の切り欠き（地の色）→ 座面 → 頭・肩・胴・腕 → 膝・すね」。胴と脚を切り離すのは脊髄損傷者に不謹慎なので、これは絶対に守る。
- **画面に説明文を置かない**: UI 部品選定規約の部品で表す。残してよい文字は、R15-04 の無効の理由と、ユーザーが違反しうる制約だけ。
- **既存の出力を変えない**: 既存の CSV の値と USB 経路は変えない。
- **コミット**: タスクごとに日本語の既存の書式（`feat(gauge): …`）で。末尾は次の 2 行。

  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01V5S8y8asg8YZQJJgUUq5Zw
  ```

- **サブエージェントを使わない**: 実装担当は自分でサブエージェントを起動しない。
- **速さ（作業者の指示「メモリをいっぱい使ってでも高速に、可能なら 1 ループあたり 30 Hz を保つ」）**:
  - ゲージは 30 Hz で届くフレームに余裕を持って追いつくこと。
  - 動かない層（地・見出し帯とアイコン・人物・溝・凡例）は、窓の大きさと DPR ごとに QPixmap に 1 度だけ描いてキャッシュする（メモリを使ってよい）。毎フレーム描くのは、帯の色・値の弧・前回・文字・見出しの右端だけにする。
  - 1 回の描画（1920×1080）は数 ms を目標にし、試験で上限を確かめる。
  - LineDemux・decode は 1 行あたり軽く（正規表現や大きなコピーを避ける）。
  - `update()` は Qt に任せてまとめる（受け取るたびに repaint を強制しない）。

## 層

`protocol`（1 行の書式と `LineDemux`。Qt に依存しない）→ `model`（値→割合・状態・段階・表示の文字。純粋）→ `scene`（800×450 の座標での弧・線・文字の列。純粋）→ `widget`（QPainter と QtSvg で scene を描くだけ）→ `window`（どの画面に出すか）。

---

### Task 1: protocol v2

**ファイル**: `app/gauge/__init__.py`（docstring 1 行だけ）、`app/gauge/protocol.py`、`tests/test_gauge_protocol.py`

**公開する API**（名前・型を変えない。向こうがこれで子の側を書く）:

```python
PREFIX = "@@GAUGE "
VERSION = 2
PART_NAMES = ("elbow_L", "elbow_R", "wrist_L", "wrist_R")
LINKS = ("waiting", "connected")
SOURCES = ("measure", "demo", "replay")

@dataclass(frozen=True)
class PartReading:
    now: float | None
    prev: float | None = None
    band: tuple[float, float] | None = None
    w1rm: float | None = None

@dataclass(frozen=True)
class GaugeFrame:
    link: str
    rep: int
    source: str = "measure"
    parts: Mapping[str, PartReading] = field(default_factory=dict)

def encode(frame: GaugeFrame) -> str    # PREFIX + JSON(区切り詰め・ASCII・数は 0.1 に丸め・非有限は null) + "\n"
def decode(line: str) -> GaugeFrame | None
```

**decode の規則**:
- 次のときは None を返す: 接頭辞が無い、JSON が壊れている、`v` が 2 でない（`"2"` の文字列も不可）、`link` が LINKS に無い、`rep` が 0 以上の int でない（bool も不可）、`source` が SOURCES に無い、`parts` が dict でない。
- 末尾の `\r\n` を受け付ける。
- 知らない鍵・知らない部位は無視する。
- 部位ごとの扱い: `now` が数でも null でもない部位は捨てる。`band` が長さ 2 の数の組で lo < hi でなければ None にする。`prev` と `w1rm` は、数でなければ None にする。

**先に書く試験**（`tests/test_gauge_protocol.py`）:
- `test_encode_writes_one_line_with_prefix_and_newline`
- `test_round_trip_restores_the_frame`（parametrize: 接続待ちで値なし／1 回目で prev=None／band=None／w1rm=None／now=None／source=replay）
- `test_decode_rejects_line_without_prefix`
- `test_decode_rejects_broken_json`
- `test_decode_rejects_other_versions`（1、3、"2"）
- `test_decode_rejects_bad_required_fields`
- `test_decode_accepts_crlf`
- `test_decode_ignores_unknown_keys_and_parts`
- `test_decode_turns_invalid_band_into_none`
- `test_decode_drops_part_with_non_numeric_now`
- `test_encode_writes_null_for_non_finite`
- `test_encode_rounds_to_one_decimal`
- `test_encode_fits_in_pipe_buf`（最悪の場合＝値が 5 桁・全部位・rep=99999・source=replay でも 512 バイト未満）
- `test_encoded_line_is_ascii`
- `test_protocol_does_not_import_qt`（別プロセスで import し、`PySide6` が `sys.modules` に無い）

**コミット**: `feat(gauge): ゲージの行の書式 v2（@@GAUGE）を足す`
**注意**: controller の指示があるまでコミットしない（土台の早送りを待つため）。

### Task 2: LineDemux

**ファイル**: `app/gauge/protocol.py` に足す、`tests/test_gauge_demux.py`

**API**: `class LineDemux`
- `feed(data: bytes) -> tuple[str, list[GaugeFrame]]`
- `flush() -> str`
- `reset() -> None`

**規則**:
- UTF-8 の増分デコーダを使う（`errors="replace"`）。多バイト文字が塊の境目で切れても化けないこと。
- 改行で終わった行: decode できたらフレーム、できなかったらそのまま（改行つきで）ログへ。
- 改行で終わっていない途中の行:
  - `PREFIX` の頭の一部、または `PREFIX` で始まる途中なら、ためておく。
  - それ以外は**すぐログへ流す**（解析スクリプトの `\r` の進捗表示を止めないため）。
- 行の途中に `PREFIX` が現れたら、その手前はログへ、`PREFIX` から先はゲージの行の候補として扱う（任意）。
- `flush()` はためている残りをログとして返す。`reset()` はバッファとデコーダを初期化する。

**試験**:
- `test_gauge_line_becomes_frame_and_not_log`
- `test_ordinary_lines_pass_through_unchanged`
- `test_gauge_line_cut_between_chunks_is_joined`（JSON の途中で切れる場合と、`"@@GA" | "UGE "` で切れる場合）
- `test_partial_ordinary_line_is_shown_immediately`
- `test_multibyte_char_cut_between_chunks_is_not_garbled`
- `test_broken_gauge_line_goes_to_log`
- `test_flush_returns_held_tail_as_log`
- `test_crlf_line_endings`
- `test_reset_forgets_previous_run`
- `test_throughput_handles_30hz_easily`（ゲージの行 3000 行と普通の行を混ぜた塊を feed し、1 行あたりの平均が 0.2 ms 未満）

**コミット**: `feat(gauge): 子の出力からゲージの行を拾う LineDemux`

### Task 3: theme

**ファイル**: `app/shell/theme.py`、`tests/test_gauge_theme.py`

- Qt に依存しない。
- 色の定数は Global Constraints の値を名前付きで持つ: `FIELD`、`TRACK`、`BAND`、`BAND_ON`、`VALUE`、`OVER`、`TEXT`、`SUBTEXT`、`HEADER`、`HEADER_SUB`、`AMBER`、`CHAIR`。
- 図柄の色は役割の色とは別の名前にする: `PLATE`、`PLATE_DARK`、`SLEEVE`。
- `contrast_ratio(a: str, b: str) -> float` は WCAG の相対輝度で計算する。
- `KNOWN_LOW_CONTRAST` には、基準に満たない組とその理由を明記する: 帯と地（約 2.66）、見出しの補足と見出しの帯（約 4.24）。

**試験**:
- `test_contrast_ratio_known_values`（白と黒で 21、同じ色で 1）
- `test_text_on_field_meets_4_5`（文字・補足・過負荷）
- `test_graphics_on_field_meet_3`（値・帯の中・過負荷・琥珀）
- `test_header_title_meets_4_5`
- `test_no_green_in_palette`（色相 90〜170° で彩度のある色が無い）
- `test_known_low_contrast_is_listed`（例外の一覧の組を除けば、すべて基準を満たす）

**コミット**: `feat(shell): 色の役割とコントラストの検査`

### Task 4: pictograms

**ファイル**: `app/shell/pictograms.py`、`tests/test_gauge_pictograms.py`

**API**:
- `figure_svg(*, figure, chair, plate, plate_dark, sleeve, background) -> str`: viewBox 0 0 800 450。`mk_subject.py` の `FIG`（最新版）をそのまま移す。
- `header_icon_svg(*, figure, plate) -> str`: viewBox 0 0 48 48。`mk_subject.py` の `ICON` の中身。

**規則**:
- SVG Tiny 1.2 の範囲で書く。`class=`、`var(`、CSS、フィルタは使わず、色は引数から属性に直書きする。
- 各要素に id を付ける（`chair-pads`、`plate-l`、`plate-r`、`sleeves`、`leg-knockout`、`seat`、`head`、`shoulders`、`torso`、`arms`、`knees`、`shins`）。
- 描く順は Global Constraints のとおり。

**試験**:
- `test_svgs_are_well_formed`（xml.etree で読める）
- `test_draw_order_is_chair_plates_knockout_seat_body`（id の出現順）
- `test_colors_come_from_arguments`（`leg-knockout` の色が `background`）
- `test_no_css_classes_or_vars`
- `test_figure_is_continuous_from_shoulders_to_feet`（QtSvg で 4 倍に描き、胴の中心の列と両すねの列を、肩から足先まで走査して、すべて人物の色＝地の色の画素が無い）
- `test_seat_passes_behind_the_legs`（座面の高さで脚の列が人物の色）

**コミット**: `feat(shell): 人物ピクトグラムの SVG（身体は一続き）`

### Task 5: model

**ファイル**: `app/gauge/model.py`、`tests/test_gauge_model.py`

**純粋な関数**:
- `fraction(v, band)`: Global Constraints の式。v が None・NaN・負なら 0。
- `status(v, band) -> Status`: `Status` は Enum で NONE / SHORT / IN_BAND / OVER。band か v が None なら NONE。
- `status_label(status)`: `"✓ 目標帯"`／`"✕ 過負荷"`／`""`。
- `joule_text(v)`: 整数に丸めた文字。
- `band_labels(band)`: `("175", "213 J")` の形。
- `PART_LABELS`: `{"elbow_L": "左 上腕", "wrist_L": "左 前腕", "elbow_R": "右 上腕", "wrist_R": "右 前腕"}`。

**状態**: `GaugeState`（frozen dataclass）
- 持つもの: `phase`（WAITING / RUNNING / DONE / FAILED）、`frame: GaugeFrame | None`、`show_joules: bool`。
- 遷移:
  - `apply_frame(state, frame)`: link=waiting なら WAITING、connected なら RUNNING。DONE / FAILED のあとに届いたフレームは捨てる。
  - `finish(state, exit_code)`: 0 なら DONE、それ以外なら FAILED。
  - `reset(show_joules)`、`with_joules(state, b)`。

**見出し**: `header(state) -> Header`
- WAITING: スピナーと「Pixel 接続待ち」
- RUNNING: (`str(rep+1)`, `"回目"`)
- DONE: `"✓ 終了 N 回"`（フレームが無ければ 0）
- FAILED: `"✕ 異常終了"`
- source=replay なら、見出しに `"▶ 再生"` の印を持たせる。

**試験**:
- `test_fraction_*`（0、帯の両端、頭打ち、NaN・負）
- `test_status_boundaries`（lo−ε は SHORT、lo は IN_BAND、hi−ε は IN_BAND、hi は OVER）
- `test_status_without_band_is_none`
- `test_status_labels_carry_symbols`
- `test_joule_text_rounds_to_integer`
- `test_band_labels`
- `test_phase_follows_link_and_finish`
- `test_frames_after_finish_are_ignored`
- `test_reset_returns_to_waiting_without_frame`
- `test_header_texts`
- `test_replay_marks_header`

**コミット**: `feat(gauge): 値→割合・状態・表示の段階`

### Task 6: scene

**ファイル**: `app/gauge/scene.py`、`tests/test_gauge_scene.py`

**純粋な場面の型**（Qt に依存しない）:

```python
Arc(cx, cy, radius, f0, f1, width, color, role, part)        # 端は常に平ら
Line(x0, y0, x1, y1, width, color, alpha, round_cap, role, part)
Run(text, size, weight, color);  Label(x, y, runs, align, role, part)
Spinner(cx, cy, r, phase)
Scene(arcs, lines, labels, spinner, show_figure=True, header_band=True)
build_scene(state, *, spinner_phase=0.0) -> Scene
qt_arc_angles(f0, f1) -> (start_deg, span_deg)   # (180 − 180·f0, −180·(f1 − f0))
Scene.find(role, part=None) -> list
```

**数値**（`mk_subject.py` の値をそのまま使う）:
- 扇の中心: elbow_L は (165,196)、wrist_L は (165,348)、elbow_R は (635,196)、wrist_R は (635,348)（鏡の対応）。
- 本体の弧: R=62、W=22。
- 帯: 半径 R+W/2+7=80、太さ 7。
- 帯の数字: 半径 94、補足色 11px。f < 0.45 は右揃え、f > 0.55 は左揃え、その間は中央揃えで上へずらす。
- 縁と値の弧: 縁は地の色で幅 W、その上に値の弧を幅 W−4（片側 2px）で描く。
- 前回の目盛り: 半径 R−W/2−3 から R+W/2+3、太さ 2.5、白で alpha 0.7、端は丸。
- 値の文字: (cx, cy−6) に 22px 800 の数字と、12px 600 の「 J」。
- 状態: (cx, cy+14) に 13px。✓ は白 700、✕ は過負荷の色 800。J がオフなら (cx, cy) に 16px。
- 部位名: (cx, cy+38) に 15px 700。
- 見出し帯: (0,0,800,64)。
- 凡例: 帯の見本は (322,424)〜(350,424) 太さ 7 と、「目標帯」(356,428)。前回の見本は x=414、y 416〜432 と、「前回」(422,428)。

**描き分け**:

| 状態 | 描くもの |
|---|---|
| WAITING | 溝と帯。J がオンなら帯の数字も。値・前回は描かない |
| RUNNING で prev が null | 値あり、前回の目盛りなし |
| RUNNING で prev あり | 値と前回の目盛り |
| DONE | 溝・帯・前回の目盛り（最後のフレームの prev）。値と状態は描かない |
| FAILED | 直前の表示のまま |
| band が null の部位 | 溝と部位名だけ（J がオンなら値の数字も） |
| now が null の部位 | 溝と帯だけ |
| J がオフ | 値と帯の数字を消し、状態を cy へ上げる |

- 目標帯の中なら、帯を帯の中の色にする。過負荷なら、値の弧と状態を過負荷の色にする。

**試験**:
- `test_dials_are_mirrored`
- `test_waiting_shows_groove_and_band_only`
- `test_first_rep_has_value_but_no_prev_tick`
- `test_nth_rep_has_value_and_prev_tick`
- `test_in_band_lights_the_band_and_keeps_white_arc`
- `test_over_draws_red_arc_and_label_with_band_unlit`
- `test_short_has_no_state_label`
- `test_value_arc_has_field_rim_of_2`
- `test_arc_ends_are_flat`
- `test_joules_off_hides_numbers_and_raises_state`
- `test_null_band_draws_groove_only`
- `test_null_now_draws_no_value`
- `test_done_shows_prev_ticks_without_values`
- `test_failed_keeps_previous_dials`
- `test_value_clamps_at_right_end`
- `test_legend_has_only_band_and_prev`
- `test_band_labels_do_not_collide_for_narrow_right_band`（帯が [175,212.5] や [112,136] のとき、文字の箱の概算が互いに重ならず、800×450 に収まり、人物の範囲 x 300〜500 に入らない）
- `test_qt_arc_angles`

**コミット**: `feat(gauge): ゲージの描く場面の計算`

### Task 7: widget

**ファイル**: `app/gauge/widget.py`、`app/core/qt.py`（QtSvg を出す 1 行が要るなら足す）、`tests/test_gauge_widget.py`

- `paint_scene(painter, scene, w, h, figure_renderer, icon_renderer)`: `s = min(w/800, h/450)` で拡大し、余白を地の色で埋め、中央に寄せて描く。
  - 弧は `QPainterPath.arcMoveTo/arcTo` と FlatCap のペン。
  - 文字は `setPixelSize`。字体の候補は Hiragino Sans → Hiragino Kaku Gothic ProN → 同梱の `assets/fonts/ipaexg.ttf`（`app.core.resources.japanese_font_path()`、`addApplicationFont`）。
  - 大きさの違う字を並べる行は `QFontMetricsF` で幅を測って並べる。
  - 人物と見出しのアイコンは `pictograms` の SVG を QSvgRenderer（色の組ごとにキャッシュ）で描く。
- `render_image(state, w, h) -> QImage`: スナップショットと試験用。
- `class GaugeWidget(QWidget)`: `set_frame`、`set_show_joules`、`finish`、`reset`、`state` のプロパティ。スピナーの QTimer（60 ms）は、WAITING で表示中のときだけ動かす。

**試験**:
- `test_renders_offscreen`
- `test_over_has_red_pixels_in_that_dial`
- `test_in_band_has_bright_blue_pixels`
- `test_waiting_has_no_white_arc_pixels`
- `test_letterbox_is_field_color`（1920×1200）
- `test_figure_is_drawn`
- `test_setters_update_state`
- `test_spinner_timer_runs_only_while_waiting`
- `test_check_and_cross_glyphs_are_available`（`QFontMetrics.inFontUcs4`）
- `test_static_layer_is_cached`（同じ大きさで 2 回描いても、動かない層の QPixmap は作り直さない。大きさが変われば作り直す）
- `test_repaint_is_fast_enough`（1920×1080 で 60 回描き、1 回の平均が 15 ms 未満。ゆるい上限で、退行だけを捕まえる）

**コミット**: `feat(gauge): 扇形ゲージの描画`

### Task 8: window

**ファイル**: `app/gauge/window.py`、`tests/test_gauge_window.py`

- `choose_screen(screens, avoid) -> QScreen | None`: avoid（作業者の窓がある画面）と違う最初の画面を返す。無ければ None。
- `class GaugeWindow(QWidget)`
  - Qt.Window、タイトルは「上肢の仕事量」、`WA_QuitOnClose=False`。中身は GaugeWidget だけで、余白は 0。部品は置かない。
  - 閉じても隠れるだけにする。
- メソッド:
  - `begin(*, show_joules, avoid_screen=None)`: reset → J を反映 → present。
  - `present(avoid_screen=None)`:
    - 別の画面がある: `winId()` で窓を作ってから `setScreen`、`setGeometry(screen.geometry())`、`showFullScreen()`。
    - 別の画面が無い: `resize(1280,720)` → 中央へ → `showNormal()`。
  - `set_frame(frame)`、`set_show_joules(b)`、`finish(exit_code)`。
- 第 2 モニタの選び方は `Gauge_display.py:675-686` を参考にする。

**試験**:
- `test_choose_screen_avoids_the_operator_screen`（偽の画面）
- `test_choose_screen_returns_none_with_one_screen`
- `test_single_screen_opens_1280x720_window`
- `test_window_does_not_keep_the_app_alive`
- `test_begin_resets_to_waiting_and_applies_joules`
- `test_has_no_controls`
- `test_closed_window_can_begin_again`
- `test_public_api_names`（`set_frame`、`set_show_joules`、`begin`、`finish` がある）

**コミット**: `feat(gauge): ゲージ窓（第 2 モニタに全画面）`

### Task 9: demo

**ファイル**: `app/gauge/demo.py`、`tests/test_gauge_demo.py`

- **シナリオ**: 名前付きの GaugeState を並べる。
  - `01_waiting`、`02_first`、`03_nth`（画面案①の値: 132/18/231/96 J。帯は v2 の実際の形）、`04_joules_off`、`05_band_null`（1 部位だけ null）、`06_over_clamp`（1.25·hi を超える）、`07_done`、`08_failed`、`09_wide_band`（帯 50〜200 J）、`10_replay`、`11_now_null`
  - 帯の例: 肘は W1RM 250 → [175, 212.5]、手首は W1RM 160 → [112, 136]
- **CLI**:
  - `python -m app.gauge.demo`: ゲージ窓を開き、合成のセット（接続待ち 2 秒 → 8 回、うち 1 回は過負荷、1 回は不足 → 終了）を QTimer で回す。
  - `--snapshot DIR [--size 1600x900]`: QApplication を作る前に `QT_QPA_PLATFORM` が未設定なら offscreen にする。状態ごとに PNG を書く。既存の QApplication があれば使う。
  - `--emit [--count N] [--interval S] [--exit-code K]`: `sys.stdout.write(encode(frame))` と flush で行を出す（source="demo"）。

**試験**:
- `test_scenario_covers_every_state`
- `test_snapshot_writes_one_png_per_state`
- `test_emit_prints_decodable_lines`（別プロセスで `--emit --count 5 --interval 0` を走らせ、5 行すべてが decode でき、ほかの行が無い）
- `test_emit_exit_code_option`

**コミット**: `feat(gauge): 合成データのデモと PNG の書き出し`

### Task 10: controls

**ファイル**: `app/shell/controls.py`、`tests/test_gauge_controls.py`

- `ToggleSwitch(QAbstractButton)`: checkable、StrongFocus。paintEvent で溝・つまみ・文字を描く。accessibleName を付ける。Space で切り替わる。
- `Disclosure(QWidget)`: QToolButton（checkable、矢印は右と下）と、中身の QWidget の setVisible。`set_open`、`is_open`、`set_badge(n)` を持つ。
- `CountBadge(QLabel)`: 0 のときは隠れる。赤地に白で「✕ n」。
- `StatusText(QLabel)`: 記号と文字の組で出す。例: `set_status("running", rep=7)` → 「● 計測中 7 回目」（琥珀の点）、「停止中」、「✓ 正常終了」、「✕ 異常終了」、`set_link(connected)` → 「● Pixel 接続」／「○ Pixel 接続待ち」。
- 緑は使わない。色は theme から取る。

**試験**:
- `test_switch_toggles_on_click_and_space`
- `test_switch_has_text_and_accessible_name`
- `test_disclosure_shows_content_when_opened`
- `test_badge_shows_count_and_hides_at_zero`
- `test_status_text_pairs_symbol_and_words`

**コミット**: `feat(shell): スイッチ・開示・状態表示の部品`

### Task 11: settings

**ファイル**: `app/core/settings.py`（CURATED の**末尾**）、`tests/test_gauge_settings.py`

- `GAUGE_SHOW_JOULES`: type bool、code_default "1"、group「表示」、`ui_visible=False`（操作口はスイッチだけ）、description は 1 行。
- `BODY_MASS_KG`: `ui_visible=True`、group「被験者」、description「体重 [kg]」。既定は効いている値の 65 のまま。

**試験**:
- `test_gauge_show_joules_is_bool_default_on`
- `test_gauge_show_joules_is_not_in_the_form`
- `test_gauge_show_joules_survives_save_and_load`
- `test_body_mass_is_visible_with_default_65`
- 既存の `tests/test_settings.py` と `tests/test_shell_smoke.py` が通ること

**コミット**: `feat(settings): J の数値の表示と体重を設定に出す`

### Task 12: worker の振り分け

**ファイル**: `app/runners/worker.py`、`app/gauge/demo.py`（`--via-worker` を足す）、`tests/test_gauge_worker.py`

- `WorkerRunner` に `gauge_frame = QtCore.Signal(object)` を足す。
- `LineDemux` を持ち、`start` で reset する。`_drain_output` は bytes を feed し、ログの文字を `output` に、フレームを `gauge_frame` に出す。
- `_on_finished` では、drain → `flush()` の残りを output → 既存の順（停止ディレクトリの削除 → 「[終了]」 → stopped → finished）。
- QProcess の exitStatus が CrashExit なら、終了コードを 0 として扱わない（0 のときは 1 にする）。
- 内部の `_handle_bytes(data)` を試験から呼べるようにする。
- `demo --via-worker [--snapshot DIR]`: 自分自身を `WorkerRunner("script")` で `--emit` 付きで起動し、`gauge_frame` と `finished` をゲージ窓へつなぐ。終わったら `frames=N log_gauge_lines=M exit=K` を出し、PNG を書く。

**試験**:
- `test_gauge_lines_reach_gauge_frame_not_output`
- `test_split_line_across_chunks`
- `test_tail_is_flushed_before_finished`（出る順: 残りのログ → stopped → finished）
- `test_new_run_starts_with_empty_buffer`
- `test_real_child_frames_arrive`（`WorkerRunner("script")` で `app.gauge.demo --emit --count 5 --interval 0` を起動し、フレームが 5 つ届き、出力に `@@GAUGE` が無い。`waitForFinished` で待つ）
- `test_crash_exit_is_reported_as_nonzero`

**コミット**: `feat(worker): ゲージの行を gauge_frame へ振り分ける`

### Task 13: 計測画面 P0（最優先）

**ファイル**: `app/shell/page_measure.py`、`tests/test_gauge_measure_page.py`

- MeasurePage は `_gauge_window` と `_run_role` を `super().__init__` より**前**に作る（`RunnerPage.__init__` が `_on_state` を呼ぶため）。
- **主ボタン**: `_main_button` の 1 つにする。押すと、`is_running` なら stop、でなければ開始。
  - `_on_state` を上書きし、starting / running では「停止」、stopped では「計測を開始」にする。無効化の一覧からは外す。
  - 今の「停止」ボタンは無くす。
- **J スイッチ**: `ToggleSwitch("J の数値")` を主ボタンの横に置く。
  - 混成（`self._runner.role == "hybrid_measure"`）のときだけ見せる。実行中も押せる。
  - 切り替えたら、`settings.set("GAUGE_SHOW_JOULES", b)` と、開いていればゲージ窓の `set_show_joules(b)`。
- **ゲージ窓の結線**:
  - 開始して `runner.start()` が真なら `_run_role = role`。混成なら `_gauge_window.begin(show_joules=設定値, avoid_screen=self.window().screen())`。
  - `runner.gauge_frame` → `_gauge_window.set_frame`。
  - `runner.finished(code)` → 混成なら `_gauge_window.finish(code)`。
  - `shutdown()` → `super().shutdown()` のあとに窓を閉じる。
  - 起動に失敗したら窓を開かない。

**試験**:
- `test_main_button_swaps_label_with_state`
- `test_main_button_stays_enabled_while_running`
- `test_main_button_starts_then_stops`
- `test_joules_switch_only_for_hybrid`
- `test_joules_switch_enabled_while_running`
- `test_switch_updates_setting_and_open_gauge`
- `test_hybrid_start_opens_gauge_window_in_waiting_state`
- `test_usb_start_does_not_open_gauge_window`
- `test_failed_start_does_not_open_gauge_window`
- `test_gauge_frames_reach_the_window`
- `test_finished_zero_shows_done_nonzero_shows_failed`
- `test_shutdown_closes_gauge_window`
- `test_restart_resets_gauge_window`

**コミット**: `feat(measure): 主ボタンの入れ替え・J スイッチ・ゲージ窓の結線`

### Task 14: 計測画面 P1a（見出しの状態）

**ファイル**: `app/shell/page_measure.py`、`tests/test_gauge_measure_page.py`

- MeasurePage では緑の `_badge` を隠し、`StatusText` を 2 つ置く。
  - 1 つめ: 実行の状態。混成なら回数も出す（「● 計測中 7 回目」）。
  - 2 つめ: 混成のときだけの Pixel 接続（点と文字の形を変え、色だけにしない）。
- 回数と接続は `gauge_frame` から取る。

**試験**:
- `test_header_shows_amber_dot_and_rep_while_running`
- `test_header_shows_link_only_for_hybrid`
- `test_header_after_finish_shows_result`
- `test_status_badge_is_hidden_on_measure_page`

**コミット**: `feat(measure): 見出しの状態を点と文字で出す`

### Task 15: 計測画面 P1b（詳細設定の開示と、説明文の削除）

**ファイル**: `app/shell/page_measure.py`、`app/shell/widgets.py`（`SettingsForm(settings, exclude=frozenset())`）、`tests/test_gauge_measure_page.py`、`tests/test_hybrid_gui.py`（計測画面の部分だけ直す）

- 「実験者用の詳細設定」の Disclosure（既定で閉じる）に、次を入れる。
  - 入力のラジオ 2 つ（「USB カメラ 2 台」「Mac＋Pixel」。QButtonGroup で role を切り替える）
  - 被験者番号（`SUBJECT_ID`）
  - 体重（QDoubleSpinBox の小数 1 桁、範囲 20〜200、直後に QLabel「kg」）
  - 「開発・診断用」の入れ子の Disclosure の中に `SettingsForm(exclude={"SUBJECT_ID","BODY_MASS_KG"})`
- 実行中は詳細設定に触れなくし、その理由「計測中は変更できません」を出す（R15-04）。
- 「デモ用…」の説明文と `_output_label` を消す。代わりに、終了後に「出力フォルダ」リンク（QLabel のリンク → `QDesktopServices.openUrl`）を出す。
  - USB は `measurement_output_dir()`、混成は `app.hybrid.paths.measurement_root()`。
  - 向こうの `tests/test_output_dir.py` がソースに `measurement_output_dir()` があることを検査しているので、その呼び出しは残す。
- `tests/test_hybrid_gui.py` の計測画面の部分をラジオで試す形に直す。キャリブレーション画面の部分は今のまま。

**試験**:
- `test_input_is_a_pair_of_radio_buttons`
- `test_radio_switches_role`
- `test_advanced_settings_are_in_a_closed_disclosure`
- `test_subject_and_body_mass_rows_edit_settings`
- `test_nested_form_excludes_dedicated_rows`
- `test_settings_disabled_while_running_with_reason`
- `test_no_explanatory_text_left`
- `test_output_folder_link_after_finish`

**コミット**: `feat(measure): 実験者用の詳細設定を開示にしまい、説明文を消す`

### Task 16: 計測画面 P2

**ファイル**: `app/shell/page_measure.py`、`app/shell/main_window.py`、`tests/test_gauge_measure_page.py`

- **校正の日時**:
  - 読み方: `app.hybrid.paths` の校正の置き場の `latest.json` を直接読み、ディレクトリ名 `%Y%m%d_%H%M%S_%f` を「2026-09-23 21:51」にする（`calibration_io` は import しない）。
  - 表示: 無ければ「未校正」。詳細設定の中に、定義の行と「変更」リンクを置く。
  - 結線: リンクで `calibration_requested` シグナルを出し、MainWindow が nav を 1 にする。
- **件数バッジ**: `app_default` を持つ 4 項目のうち、既定と違う値で有効になっているものの数を、開示の見出しに出す。
- **ログ**: `setPlaceholderText("未実行")`。
- **即時保存**: スイッチの切り替えで `settings_edited` シグナルを出し、MainWindow が保存する。

**試験**:
- `test_calibration_time_is_read_from_latest_json`
- `test_change_link_requests_calibration_page`
- `test_badge_counts_enabled_broken_flags`
- `test_log_shows_placeholder_before_first_run`
- `test_switch_saves_immediately`

**コミット**: 項目ごと

### Task 17: 凍結の準備と設計書の更新

**ファイル**: `packaging/app.spec`、`docs/superpowers/specs/2026-09-23-subject-gauge-design.md`、`tests/test_cross_platform.py`（一覧の**末尾**に `app.gauge.protocol`・`app.gauge.model`・`app.gauge.scene` を足す）

- `packaging/app.spec` の `hiddenimports` に `"app.gauge.demo"` を足す。
- 設計書を更新する:
  - §6.3 を v2 の形式にする。
  - §9 の閾値を「子が論文の W_0.70 と W_0.85 を計算して band で送る」にする。
  - 状態の境界（lo ≤ v < hi）、`scene.py` と `controls.py` の追加、`tracker` と `thresholds` は向こうの持ち分であること、source=replay を書き足す。

**コミット**: `build(packaging): ゲージのデモを凍結版に入れ、設計書を合意に合わせる`

### Task 18: .app の確認（controller が行う）

1. `PYTHON=$PY ./packaging/build_macos.sh` でビルドする。
2. `HOME=<scratch> QT_QPA_PLATFORM=offscreen $APP --role script --module app.gauge.demo --snapshot <絶対パス>` で、凍結版で PNG を書き出して目視する。
3. `--via-worker` で子の標準出力が GUI に届くことを確かめる（frames>0、ゲージの行はログに 0 件、exit=0）。
4. 凍結版の GUI を起動し、計測画面を offscreen で撮る。
5. 目視: 人物に切れ目が無い、文字が重ならない、✓ と ✕ と日本語が出ている。
6. PNG を作業者に送る。
