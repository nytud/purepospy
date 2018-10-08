# PurePosPy

A Python3 wrapper for [PurePos](https://github.com/ppke-nlpg/purepos).

## Requirements

See requirements.txt.

## Usage

This repository contains the following classes and utilities:

- PurePOS: a Python 3 wrapper class around PurePos, which can be used for training and tagging
- Tokenizer: a simple rule-based sentence splitter and tokenizer class for Hungarian
- RawToPurePOS: a class allows to feed raw text to PurePos after tokenization
- PurePOSTCPHandler: a simple TCP server variant of RawToPurePOS
- read_data_w_annotation: a utility function to read data from string to sentences and annotations separated
- put_data_together: a utility function to zip sentences and annotations (ex. for feeding to PurePOS as training data)

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
PurePos has its own license.
