# PurePOSPy

A Python3 wrapper for [PurePOS](https://github.com/ppke-nlpg/purepos). (A previous version of the code is stored at https://github.com/ppke-nlpg/purepospy .)

## Requirements

- See requirements.txt.
- Model (already included) e.g. [Szeged corpus](http://rgai.inf.u-szeged.hu/index.php?lang=en&page=SzegedTreebank) with [emMorph tags](http://e-magyar.hu/en/textmodules/emmorph_codelist).
- Run `make download-purepos` to get PurePOS before first use or building the package.

## Install on local machine

  - Install [git-lfs](https://git-lfs.github.com/)
  - `git-lfs install`
  - Clone the repository: `git clone https://github.com/dlt-rilmta/purepospy` (It should clone the model file also!)
  - ``sudo apt install `cat Aptfile` ``
  - `make build`
  - `sudo pip3 install dist/*.whl`
  - Use from Python

## Usage

The authors recommend using PurePOSPy in [emtsv](https://github.com/dlt-rilmta/emtsv) the new version of [e-magyar](http://www.e-magyar.hu) language processing system. This module is called *emTag*.

The PurePOS class is a Python 3 wrapper class around PurePOS, which can be used for training and tagging

	```python
	>>> from purepospy import PurePOS
	>>> p = PurePOS('szeged.model')  # New, or existing file
	>>> tok = ['word', 'lemma', 'tag']
	>>> sent = [tok, tok, ...]
	>>> sentences = [[sent],[sent], ...]
	>>> p.train(sentences)  # Training, optional
	>>> p.tag_sentence('Sentence as string , tokenised .')
	Output#output_lemma#output_tag as#as_lemma#as_tag string#string_lemma#string_tag .#.#PUNCT
	```

## License

This Python wrapper, and utilities are licensed under the LGPL 3.0 license.
PurePOS has its own license.
