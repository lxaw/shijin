# shijin 詩人
## A small language model for writing nice Japanese.

## What is it? 
- A language model trained on [AozoraBunko](https://www.aozora.gr.jp/).
- Datasource: https://huggingface.co/datasets/globis-university/aozorabunko-clean

## How big?
- 21.594913 M parameters at the moment.

## Examples?
- See `\example_scripts` for example scripts of this model.
- See `\example_text` for example text generated from this model.

## To-Do:
- As I am but a puny university student, I do not have access to big GPU.
- As such, let's see if we can use transfer learning from a LLM for English!
- Languages, even though they may look and sound different, all share some inner structure.
- As such, perhaps we can get good results from transfer learning?

## Thanks!
- This code is *heavily* inspired from [karpathy](https://github.com/karpathy/ng-video-lecture)'s YouTube series. The guy is a legend. And handsome!