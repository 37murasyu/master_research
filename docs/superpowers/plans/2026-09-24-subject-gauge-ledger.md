# 被験者ゲージ実装の台帳（controller の記録。クラウドの統合担当への引き継ぎ用）

ローカルの `.superpowers/sdd/2026-09-24-subject-gauge/progress.md`（git 管理外）を、クラウドから読めるようにそのまま写したもの。
`Ruling N:` の行が controller の判断（何を決めたか — なぜ — 誤っていたときの代償）。

# SDD ledger — plan: docs/superpowers/plans/2026-09-24-subject-gauge.md

Spec: docs/superpowers/specs/2026-09-23-subject-gauge-design.md（Global Constraints が §6.3・§9 より優先）
Base: a6abf3a（向こうの HEAD が届いたら ff-only で進めてから Task 1 をコミット）

## Preflight scan

| 組／タスク | 生産 → 消費 | 所見 |
|---|---|---|
| T1 ↔ T2 | protocol.py の型 → LineDemux が decode を使う | 同じファイルへの追記。整合 |
| T1 → T5 | GaugeFrame/PartReading → GaugeState | 整合（now: float|None を model が None 扱い） |
| T5 → T6 | fraction/status/header → build_scene | 整合 |
| T3,T4,T6 → T7 | 色・SVG・Scene → paint_scene | 整合 |
| T7 → T8 → T9 | GaugeWidget → GaugeWindow → demo | 整合 |
| T9 ↔ T12 | demo.py を T12 が追記（--via-worker） | 順番どおりなら衝突なし |
| T10 → T13/T14/T15 | ToggleSwitch/Disclosure/StatusText/CountBadge | 整合 |
| T11 → T13 | GAUGE_SHOW_JOULES | 整合 |
| T12 → T13 | gauge_frame シグナル | 整合 |
| T13 ↔ T14 ↔ T15 ↔ T16 | page_measure.py を順に上書き | 逐次なので衝突なし |
| T15 | widgets.SettingsForm に exclude を足す／test_hybrid_gui.py を直す | 既存の試験を壊さない範囲で |
| T17 | app.spec・設計書・test_cross_platform（末尾） | 整合 |
| T1 自身 | 試験と API | 整合 |
| T4 自身 | 描く順と FIG | mk_subject.py の FIG（v4）と同じ順。整合 |
| T6 自身 | 縁 2px（W−4）と画面案（W−6＝3px） | Ruling 1 |
| T7 自身 | 速さの試験（平均 15 ms 未満） | offscreen の raster で環境差あり → ゆるい上限で退行だけ捕まえる |

- Ruling 1: 値の弧の縁は片側 2px（幅 W−4）にする — 計画と承認済みの計画書が 2px を指定しており、画面案の 3px は試作の値 — 誤りなら見た目がわずかに細くなるだけで、定数 1 つで戻せる。
- Ruling 2: 向こうの HEAD が 02:30 までに届かなければ、a6abf3a の上で Task 1 をコミットして送る（向こうが cherry-pick）— 取り決め 1 の代替 — 誤りなら取り込み時に向こうの手間が 1 つ増える。
- Ruling 3: 速さの指示（30 Hz）は、UI 側では描画のキャッシュと処理速度の試験で満たし、計測ループ本体は向こうへ依頼した — 持ち分の取り決めに従う — 誤りならループの速さは向こうの作業次第になる。

## Progress
Task 1: dispatched implementer (BASE a6abf3a, 未コミット指示。向こうの HEAD を待って ff 後にコミット)
Base: ff-only で 15a2654 へ（向こうの 8 コミット: 3a73b61..15a2654）
Task 1: committed 6a4696d, sent to peer（計画書 c94f6b2）
Task 1: review 1 — Needs fixes（Important: decode が NaN/Infinity を通す）＋実装担当の懸念（v=2.0 を通す）→ fix round 1 を依頼（FIX_BASE 6a4696d）
- Ruling 4: ⚠️「CURATED・test_cross_platform への追加は Task 1 で要るか」→ 不要。Task 11 と Task 17 で末尾に足す — 計画のタスク分けどおり — 誤りなら Task 1 の段階で一覧に載らないだけ。
Task 1: fix round 1/5 (2 件修正済み、再レビュー待ち; commits c94f6b2..888d5df)
Task 1: fix round 1/5 (2 addressed, 0 open; commits c94f6b2..888d5df)
Task 1: complete (commits 15a2654..888d5df, review clean after round 1)
Task 2: dispatched implementer (BASE 888d5df)
Task 2: implementer DONE (888d5df..59ea1bd)、review dispatched
Task 2: minor (deferred): 行の途中に出る PREFIX が塊の境目でさらに割れる二重の縁の場合に、そのフレームがログへ流れる（任意要件の縁。protocol.py:106-118）
Task 2: complete (commits 888d5df..59ea1bd, review clean)
Task 3: dispatched implementer (BASE 59ea1bd)
（01:5x〜06:22 使用量の上限で中断。Task 3 の実装担当は緑の直前で停止 → 06:22 に呼び戻し）
- Ruling 5: 残りの順番を「実機でゲージ窓が動く最短の経路」優先に組み替える: 3 → 5 → 6 → 7 → 8 → 11 → 12 → 13 → 4 → 10 → 9 → 14 → 15 → 16 → 17 → 18 — 別セッションの依頼と、作業者の目標「朝に実機でゲージが反応」に合わせる — 誤りなら人物の描画と作り込みが後回しになり、朝の時点で中央の人物が無い版になりうる（widget は figure を任意にしておく）。
Task 3: implementer DONE (59ea1bd..dd9d6cc)、review dispatched
- Ruling 6: 朝までの時間のため、読むだけのレビュー担当と次のタスクの実装担当を並行させる（実装担当は常に 1 人）— 修正が要るときは、実装中のタスクが終わってから修正を出す — 誤りなら修正の反映が 1 タスク分遅れる。
Task 5: dispatched implementer (BASE dd9d6cc)
Task 3: minor (deferred): 緑の判定の彩度のしきい値 0.25 の理由がコメントに無い（tests/test_gauge_theme.py:243）／_to_rgb01 の ValueError の分岐が試験されていない（theme.py:124-132）
Task 3: complete (commits 59ea1bd..dd9d6cc, review clean)
Task 5: implementer DONE (dd9d6cc..c2f22b4)、review dispatched
Task 6: dispatched implementer (BASE c2f22b4)
Task 5: ⚠️（コミット本文の末尾 2 行）→ controller が確認
Task 5: minor (deferred): status() が NaN を OVER と判定しうる（fraction は NaN を 0 扱い。model.py:105-117）
Task 5: complete (commits dd9d6cc..c2f22b4, review clean)
- Ruling 7: 888d5df と dd9d6cc のコミット本文で末尾 2 行のうち 1 行が欠けている。ハッシュを送った後は履歴を書き換えない約束なので直さない — 帰属の記録が 2 件だけ不完全 — 誤りでも機能への影響は無い
Task 6: implementer DONE (c2f22b4..24bb664)、review dispatched
- Ruling 8: 部位名のラベルはどの局面でも描く — 画面案（mk_subject.py の gauge()）が無条件に描いており、constraints の「溝と帯だけ」は値の表示の話と読む — 誤りなら部位名の表示を条件付きにする 1 行の変更で戻せる。
- Ruling 9: 帯の数字を中央揃えにするときの持ち上げ量 CENTER_LABEL_LIFT=10 は仮の値とし、widget の描画を目視して決める — 仕様に数値が無い — 誤りなら数字の位置が数 px ずれるだけ。
- Ruling 10: widget は人物を描くので、Task 4（pictograms）を Task 7 の前に回す（順番 3→5→6→4→7→8…）— 依存の順 — 誤りならなし。
Task 4: dispatched implementer (BASE 24bb664)
Task 6: review — Important（plan-mandated）: 値の弧の縁が画面案 mk_subject.py（W−6、片側 3px）と違い W−4（片側 2px）
- Ruling 11: 縁は片側 2px（W−4）のまま。設計書 §4「値の弧の周りに地の色で 2px の縁を付けて分離する」が正本で、画面案の 3px は試作の値（Ruling 1 と同じ）— 誤りなら VALUE_RIM_INSET を 6 にする 1 行で戻せる。
Task 6: minor (deferred): 帯の数字の中央揃えの分岐と CENTER_LABEL_LIFT が試験で通らない（scene.py:251-264）。値 10 は widget の目視で確定する
Task 6: complete (commits c2f22b4..24bb664, 1 finding ruled)
Task 4: implementer DONE_WITH_CONCERNS (24bb664..4ca086f; 懸念: 座面の試験が色の厳密一致)、review dispatched
Task 7: dispatched implementer (BASE 4ca086f)
Task 4: review — Needs fixes（Important: tests/test_gauge_pictograms.py:134 の module 直下の importorskip が、PySide6 の無い環境で Qt に依存しない 4 件までファイルごとスキップする）→ Task 7 の実装担当が終わってから fix round 1 を出す（Ruling 6）
Task 4: minor (deferred): figure_svg と header_icon_svg の <svg> の組み立てが小さく重複（pictograms.py:86-91, 113-118）
Task 7: implementer DONE (4ca086f..270042c; 描画平均 2.5ms、controller が PNG を目視: 人物に切れ目なし・文字の重なりなし)、review dispatched
Task 4: fix round 1 dispatched（resume、FIX_BASE 270042c）
Task 4: fix round 1 implementer DONE (270042c..ddfcad1)、re-review dispatched
Task 8: dispatched implementer (BASE ddfcad1)
Task 4: fix round 1/5 (1 addressed, 0 open; commits 270042c..ddfcad1)
Task 4: complete (commits 24bb664..4ca086f + ddfcad1, review clean after round 1)
Task 7: review — Approved だが Important 1: GaugeWidget.paintEvent（静止層のキャッシュ＋動く層の合成）の画素を確かめる試験が無い（レビュー担当は grab と render_image の画素一致を個別に確認済み）→ Task 8 の実装担当が終わってから fix round 1（resume）
Task 7: minor (deferred→fix round 1 で一緒に): 「静止 role と動く role は画面上で重ならない前提」をコメントで明記
Task 7: ⚠️ 高 DPI（dpr≠1）の実機での見た目は未確認 → Task 18（.app の確認）で実機のディスプレイで撮る
Task 8: implementer DONE (ddfcad1..493adfe; 懸念: 全画面の分岐は offscreen で通らない)、review dispatched
Task 7: fix round 1 dispatched（resume、FIX_BASE 493adfe）
Task 8: minor: 診断用の gauge プロパティが公開の口を 1 つ増やす（window.py:86-89）
- Ruling 12: gauge プロパティは残す — 別セッションの grab() の道具が中の状態を確かめるのに使えると既に伝えてある — 誤りなら窓の外から GaugeWidget を直接触れる余地が残るだけ。
Task 8: ⚠️ 全画面（第 2 モニタ）の経路は offscreen で通らない → Task 18 と実機で確認
Task 8: complete (commits ddfcad1..493adfe, review clean)
Task 7: fix round 1 implementer DONE (493adfe..3126d81)、re-review dispatched
Base: 向こうの fix-left-right-dynamics（272c85d、HYBRID_* の設定を含む）を merge（67b0a06、衝突なし、全体 1338 passed）
Task 11: dispatched implementer (BASE 67b0a06)
Task 7: fix round 1/5 (2 addressed, 0 open; commits 493adfe..3126d81)
Task 7: complete (commits 4ca086f..270042c + 3126d81, review clean after round 1)
- Ruling 13: 作業者の「もっと早く」に応えて、触るファイルが重ならないタスクを別々の作業ツリーで並行して実装し、controller が subject-gauge へ merge する（../mr-gauge-t9・t10・t12・t17、ブランチ murayama/gauge-t9 など、土台 67b0a06）— リモートのクレジットは push と環境構築が要るので使わない — 誤りなら merge の手間が増えるだけ（ファイルは重ならない）。
- Ruling 14: Task 12 の demo --via-worker と demo を子に使う試験は、demo が合流してから小さく足す（今回は python -c の小さな子で確かめる）— 並行のため — 誤りなら凍結版の「子の標準出力→GUI」の確認が後回しになる。
Task 10: dispatched implementer（../mr-gauge-t10、BASE 67b0a06）
Task 9: dispatched implementer（../mr-gauge-t9、BASE 67b0a06）
Task 12: dispatched implementer（../mr-gauge-t12、BASE 67b0a06）
Task 17: dispatched implementer（../mr-gauge-t17、BASE 67b0a06）
Task 11: implementer DONE (67b0a06..5d43e47)、review dispatched
（push は 403: ms-arcana に 37murasyu/master_research への書き込み権限が無い → 作業者の対応待ち。クラウドは保留）
Task 11: complete (commits 67b0a06..5d43e47, review clean)
Task 12: implementer DONE（../mr-gauge-t12、67b0a06..f03a345）→ subject-gauge へ merge、review dispatched
Task 10: implementer DONE（../mr-gauge-t10、67b0a06..c5196f5）→ subject-gauge へ merge、review dispatched
Task 17: implementer DONE（../mr-gauge-t17、67b0a06..63d4e84）→ subject-gauge へ merge
Task 12: complete（f03a345、merge 5d3f099）— review Approved。Minor parked: worker.py 冒頭 docstring の「将来 JSON Lines を足す」が古い（向こうとの衝突を避けて据え置き。最終レビューでまとめて直す）
Task 10: complete（c5196f5、merge 570013b）— review Approved。
Ruling 15: 「● Pixel 接続」の点は琥珀のまま — 琥珀は「いま生きている信号」の色で、接続も計測と同じく生きている状態。文字で区別できる — 誤りなら controls.py の 1 行の色替え
Minor parked: StatusText の知らない state は「停止中」に落とす（呼び出し側のバグを隠しうる。最終レビューで判断）
Task 13〜16: クラウドのエージェントへ（土台 e445ffe、ブランチ murayama/gauge-measure-page、タスクごとに push）
Task 9: implementer DONE（../mr-gauge-t9、67b0a06..5e73d48）→ subject-gauge へ merge、review dispatched
Fix（controller）: cefb8f5 scene の DONE で now=null の部位の前回の目盛りが消える不具合（デモの目視で発見、試験 test_done_keeps_prev_tick_when_last_now_is_null）
Task 9: complete（5e73d48、merge fda0368）— review Approved。Minor parked: SCENARIOS を import 時に作る（--emit では不要。軽いので据え置き）
Task 17: complete（63d4e84、merge e445ffe、fix 057086a）— review Needs fixes → controller が直した。
Ruling 16: Task 17 review の Important 1（worker.py の gauge_frame は未実装）は棄却 — レビューは Task 12 取り込み前の mr-gauge-t17 を見ていた。subject-gauge では 5d3f099 で実装済み — 誤りなら設計書の 1 行
  Important 2（thresholds の依存）・3（band null で J オンの中央の数字）と、見落とされていた controls の持ち分の取り違えを修正
.app: ビルド成功（fda0368+cefb8f5）。凍結版の --snapshot 11 枚が開発版と画素一致
Task 12 残り（controller）: 071c7e5 demo --via-worker と --emit の締め切り刻み（開発版 90/90・漏れ 0・30.0 Hz）
Task 13: cloud DONE（e445ffe..151e5a4: 6468499 本体、151e5a4 修正）、review dispatched
.app（071c7e5）: 凍結版 --via-worker 90/90・漏れ 0・30.0/29.9 Hz・exit 3 の素通し OK
Task 13: complete（cloud 6468499+151e5a4）— review Approved。Minor parked: J スイッチは窓が閉じていても set_show_joules を呼ぶ（安全）／開始後の役割切替の試験なし（実行中はラジオが無効なので経路が無い）
Task 14: cloud DONE（4dab0ed..c35623f）、review dispatched
Minor（Task 12）解消: d5e1754 worker.py の docstring
Task 15: cloud DONE（e472725）— murayama/gauge-measure-page を subject-gauge へ merge
Task 16: cloud で仕上げ。校正の日時と「変更」リンク（2165f45、止まった実装担当の WIP をそのまま採用）に、件数バッジ・ログの「未実行」・J スイッチの即時保存を足した
- Ruling 17: 件数バッジは「実験者用の詳細設定」と入れ子の「開発・診断用」の両方の見出しに出す — 設計書 §5.2 は外側の開示の見出しを指すが、該当の設定は入れ子の中にあり、入れ子が閉じていても場所がわかるように — 誤りなら _refresh_broken_flags の 1 行を消すだけ
- Ruling 18: 即時保存は計画どおり J スイッチだけ（ほかの欄は従来どおり窓を閉じるときに保存）— 計画書 Task 16 の範囲 — 誤りなら他の欄の変更で settings_edited を出す数行
Test（cloud・Linux）: 1400 passed / 12 failed。12 件は変更前の HEAD でも同じく落ちる環境差（mediapipe 無し、GUI 無しの cv2、フォント差での widget の画素比較）
残り: Task 18（.app の確認）は Mac で controller が行う
Task 14: review（cloud）— Approved。finished より先に state "stopped" が出る順（worker.py:194-195）なので、✓／✕ が停止中に上書きされないことを確認
Task 15: review（cloud）— Important 1: 保存済みの体重が 20〜200 の外だと、欄は丸めた値を見せるのに設定は元の値のまま子へ渡る → 結線してから値を入れ、設定を欄の値にそろえて修正（試験 test_body_mass_setting_matches_what_the_field_shows）
Task 16: review（cloud）— Approved
Minor（parked）を解消:
  - StatusText の知らない state は ValueError（黙って「停止中」にしない）
  - status() は NaN を NONE にする（fraction と同じ扱い）
  - 帯の数字の中央揃えの分岐と CENTER_LABEL_LIFT の試験を追加（値 10 は据え置き。目視は Task 18）
  - LineDemux: 行の途中の PREFIX が塊の境目で割れても、末尾の PREFIX の頭をためてフレームとして拾う
  - pictograms の <svg> の組み立てを _svg に 1 つにした
  - 緑の判定の彩度 0.25 の理由をコメントに書き、_to_rgb01 の ValueError の試験を追加
書体（作業者の依頼）: app/gauge/fonts.py を追加。Mac にアクティベート済みのフォントワークスの書体を名前で探して使う（.app には同梱しない）
- Ruling 19: 書体の組は設定 GAUGE_FONT_PRESET（rodin 既定・tsukushi・kaimin・system）。マティス EB は作業者の指示で使わない — 見出しのゴシック案（ロダン＋UD角ゴ_ラージ）を既定にした — 誤りなら code_default を変えるだけ
- Ruling 20: 役は 3 つ（見出し＝header_title、数字＝value_text・header_rep・band_label、文字＝残り）。数字には tnum を指定（Qt 6.7 以上）
- 見つからない書体はヒラギノ（見出しが明朝の組はヒラギノ明朝）→ IPAex。実機での見え方は Task 18 で `--role script --module app.gauge.fonts` と `app.gauge.demo --snapshot DIR --fonts all` で確かめる
