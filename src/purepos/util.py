"""
Copyright (C) György Orosz
Author: György Orosz <oroszgy@itk.ppke.hu>
URL: <http://github.com/ppke-nlpg/purepos>
For license information, see LICENSE.TXT
"""

from io import StringIO

def triplet2txt(triplet):
        return "#".join(triplet)
    
def txt2triplet(txt):
    return tuple(txt.split("#"))
    
def annot2txt(annot):
    token = annot[0]
    anals = annot[1]
    txt = StringIO()
    txt.write(token)
    if anals:
        txt.write("{{")
        for anal in anals:
            if len(annot)>2 and not annot[2]:
                txt.write("*")
            txt.write(anal[0])
            txt.write(anal[1])
            if len(anal)>2 and anal[2]:
                txt.write("$$")
                txt.write(str(anal[2]))
            txt.write("||")
        txt.write("}}")
        out = txt.getvalue()[:-4]+"}}"
    else:
        out = txt.getvalue()
    return out
    
def parse_text(txt):
    lines = txt.strip().split("\n")
    return [ [ (txt2triplet(tok)) for tok in line.split()] for line in lines]

def parse_scoredsent(txt):
    s_pos = txt.find("$$")
    score = float(txt[s_pos+2:-2])
    rest = txt[:s_pos]
    tokens = rest.split(" ")
    tokens = list(map(txt2triplet, tokens))
    return (tokens, score)