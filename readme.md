# shijin 詩人
# 詩人のための小規模言語モデル

![irasutoya poet](https://github.com/lxaw/shijin/blob/main/git_assets/music_ginyuu_shijin.png)

## A small language model for writing nice Japanese.
### Issueを英語で書く必要はありません。

## What is it? 
- A language model trained on [AozoraBunko](https://www.aozora.gr.jp/).
- Datasource: https://huggingface.co/datasets/globis-university/aozorabunko-clean
## なんだこれ？
- [青空文庫](https://www.aozora.gr.jp/)をベースにした小規模言語モデルです。
- この[Qiita記事](https://qiita.com/akeyhero/items/b53eae1c0bc4d54e321f)でデータセットの詳細が書いてあるので、ご参考まで。

## How big?
- 21.693217 M parameters at the moment.
## 大きいの？
- 現時点で21,693,217個のパラメーターです。ちっちゃなモデルではありますが、侮らないでね。もしかして次の芥川賞を取るかもしれんぞ！

## Examples?
- See `\example_scripts` for example scripts of this model.
- See `\example_text` for example text generated from this model.
## どうやって走らせるの？
- このモデルの例題スクリプトは`\example_scripts`にあります。
- このモデルから生成された例文は`\example_text`にあります。

## To-Do:
- As I am but a puny university student, I do not have access to big GPU.
- As such, let's see if we can use transfer learning from a LLM for English!
- Languages, even though they may look and sound different, all share some inner structure.
- As such, perhaps we can get good results from transfer learning?
## まだやっていないこと：
- 一文なしの大学生の私には、数多くの高性能なGPUは周りに転がっていません。
- そのため、英語のLLMからの転移学習を使用できるかどうかを見てみましょう！
- 言語は見た目や音が異なるかもしれませんが、ある程度の内部構造を共有しているはずです。
- なので、転移学習から良い結果を得ることができるかもしれません。

## Thanks!
- This code is *heavily* inspired from [karpathy](https://github.com/karpathy/ng-video-lecture)'s YouTube series. The guy is a legend. And handsome!
## ありがとう！
- このコードは、[karpathy](https://github.com/karpathy/ng-video-lecture)氏のYouTubeシリーズから*大いに*インスピレーションを受けました。神。



