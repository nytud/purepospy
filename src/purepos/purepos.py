#!/usr/bin/env python3

"""
Copyright (C) György Orosz
Author: György Orosz <oroszgy@itk.ppke.hu>
URL: <http://github.com/ppke-nlpg/purepos>
For license information, see LICENSE.TXT

A Python 3.x interface for PurePos.
"""

from io import StringIO
import os
import sys
from subprocess import Popen, PIPE

from .util import triplet2txt, annot2txt, txt2triplet, parse_scoredsent


_purepos_encoding = "UTF-8"
_purepos_version = "2.0"
_purepos_bin = "purepos-" + _purepos_version +".one-jar.jar"

_base_dir = os.path.dirname( os.path.abspath( __file__) )
_purepos_dir = _base_dir
sys.path.append(_base_dir)


class PurePosBase(object):
    
    def __init__(self, model_path, mode, options=None,
                 encoding = None, verbose=False):    
        self._encoding = encoding or _purepos_encoding
        self.verbose = verbose
        _options = options or []
        
        self._purepos_bin = os.path.abspath(os.path.join(_purepos_dir, _purepos_bin))
        self._cmd = ["java", "-Djava.library.path="+_base_dir, "-Dhumor.path="+_base_dir+"",
        "-jar", self._purepos_bin, mode, "-m", os.path.abspath(model_path)] + _options

        if verbose: pstderr = None
        else: pstderr = open(os.devnull, 'wb')
        
        self._purepos = Popen(self._cmd, shell=False, stdin=PIPE, stdout=PIPE,
                                stderr=pstderr)
        self._closed = False

    def __del__(self):
        self.close()

    def close(self):
        if not self._closed:
            self._purepos.communicate()
            self._closed = True

    def __enter__(self):
        return self
        
#    def __exit__(self, exc_type, exc_value, traceback):
#        self.close()
        

class PurePosTrainer(PurePosBase):
    
    def __init__(self, model_path, train_text=None, 
                 encoding=None, verbose=False):
        PurePosBase.__init__(self, model_path, "train", [], encoding, verbose)
        if train_text:
            self.train(train_text, finalize=True)
            
    def train(self, text, finalize=True):
        self._train(text)
        if finalize:
            self.finalize_training()
            
    def _train(self, sents):
        """
        Text is a list of sentences. A sentence is a list of (word, lemma, tag) tuples.
        """
        
        txt = "\n".join([ " ".join(map(triplet2txt, sent)) for sent in sents])

        self._purepos.stdin.write(txt.encode(self._encoding))
        self._purepos.stdin.write("\n".encode(self._encoding))
        self._purepos.stdin.flush()
    
    def finalize_training(self):
        self.close()
            
class PurePosTagger(PurePosBase):

    def __init__(self, model_path, multi_tag=None, encoding=None, verbose=False):
        options = []
        self._multitag=bool(multi_tag)
        if multi_tag:
            options =  ["-n", str(multi_tag), "-d"]
        PurePosBase.__init__(self, model_path, "tag", options, encoding, verbose)
        


    def tag(self, tokens):
        """
        Tags a single sentence: a list of words.
        The tokens should not contain any newline characters.
        """
        out = StringIO()
        assert type(tokens) is list
        for token in tokens:
            token_str = token
            if isinstance(token, tuple):
                token_str = annot2txt(token)
                
#             if isinstance(token_str, unicode):
#                 token_str = token_str.encode(self._encoding)
            elif isinstance(token_str, str):
                pass
            else: 
                raise Exception("Unkwon input format: %s"%str(token))
            assert "\n" not in token, "Tokens should not contain newlines"
            out.write(token_str)
            out.write(" ")
                
        out = out.getvalue()[:-1]
        try:
            self._purepos.stdin.write(out.encode(self._encoding))
            self._purepos.stdin.write("\n".encode(self._encoding))
            self._purepos.stdin.flush()
            out = self._purepos.stdout.readline().decode(self._encoding).strip()
        except:
            sys.stderr.write(repr(out) + "\n")
            raise
        if not self._multitag:
            ret = map(txt2triplet, out.split(" "))
        else:
            sents = out.split("\t")
            ret = map(parse_scoredsent, sents)
            
        return list(ret) 

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file")
    args = parser.parse_args()
    p = PurePosTagger(args.model_file, verbose=False)
    print('Enter one sentence per line to tag them.')
    while True:
        inp = sys.stdin.readline().strip()
        print(p.tag(inp.split()))
        sys.stdout.flush()

if __name__ == "__main__":
    main()
