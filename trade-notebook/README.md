# 売買シナリオ帳

銘柄ごとの分析、エントリーゾーン、イベント日程、売買記録、メモを保管するページです。

- 公開先: https://claude.ai/artifact/SWFgRow2jr64hawDtCr6j8
- データは Artifact の `db`（`stocks` / `events` / `trades` / `notes` コレクション）に保存されます。
- 通知は Claude Code の Routine（決算・日銀会合・権利落ちなどの日付と、毎週金曜の週次チェック）から届きます。Routine は判定結果を `notes` に追記し、`stocks/<code>` の株価を更新します。
