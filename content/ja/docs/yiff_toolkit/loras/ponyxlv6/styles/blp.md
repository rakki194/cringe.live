---
weight: 1
bookFlatSection: false
bookToC: false
title: "blp"
summary: ""
---

<!--markdownlint-disable MD025 MD033 -->

# blp-v1e400

---

## はじめに

---

40個もの異なるカスタムノードを使用してシグマやノイズ注入を変更することなく、[blp](https://e6ai.net/posts?tags=blp)の独特なAIアートスタイルを再現します。より写実的な仕上がりを求める場合は、CFGを`6.0から9.0`の間に設定し、サンプラーとスケジューラには`DPM++ 2M Karras`を使用することをお勧めします。また、よりカートゥーン調/夢のような生成を求める場合は、CFGを`6.0`の低めに設定して`Euler a`を使用することもできます。`1536x1536`での二次パスは、`Euler a`でCFG `9.0`、デノイズを`0.4`から`0.69`の間に設定することをお勧めします。

このLoRAをネガティブウェイト`-0.5`で使用すると、生成物にセピア調が加わるという報告があります。

## コンテンツ

---

[⬇️ LoRAダウンロード (229MB)](https://huggingface.co/rakki194/yt/resolve/main/ponyxl_loras/blp-v1e400.safetensors?download=true)

[⬇️ 圧縮版LoRAダウンロード (42.1MB)](https://huggingface.co/rakki194/yt/resolve/main/ponyxl_loras_shrunk_2/blp-v1e400_frockpt1_th-3.55.safetensors?download=true)

[🖼️ メタデータ付きサンプル画像](https://huggingface.co/k4d3/yiff_toolkit/tree/main/static/{})

[📐 データセット](https://huggingface.co/datasets/k4d3/furry/tree/main/by_blp)

[📊 メタデータ](https://huggingface.co/k4d3/yiff_toolkit/raw/main/ponyxl_loras/blp-v1e400.json)

## プロンプトガイド

---

キーワード：

- `blp`

### 推奨タグ

```md
detailed background, amazing_background, scenery porn, feral,
```

---

<!--
HUGO_SEARCH_EXCLUDE_START
-->
{{< related-posts related="docs/yiff_toolkit/loras/ponyxlv6/styles/darkgem/ | docs/yiff_toolkit/loras/ponyxlv6/styles/honovy | docs/yiff_toolkit/loras/ponyxlv6/styles/jinxit" >}}
<!--
HUGO_SEARCH_EXCLUDE_END
-->
