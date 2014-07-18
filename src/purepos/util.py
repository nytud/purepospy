"""
Copyright (C) György Orosz
Author: György Orosz <oroszgy@itk.ppke.hu>
URL: <http://github.com/ppke-nlpg/purepos>
For license information, see LICENSE.TXT
"""

from io import StringIO

def triplet2txt(triplet):
        return u"#".join(triplet)
    
def txt2triplet(txt):
    return tuple(txt.split(u"#"))
    
def annot2txt(annot):
    token = annot[0]
    anals = annot[1]
    txt = StringIO()
    txt.write(token)
    if anals:
        txt.write(u"{{")
        for anal in anals:
            if len(annot)>2 and not annot[2]:
                txt.write(u"*")
            txt.write(anal[0])
            txt.write(anal[1])
            if len(anal)>2 and anal[2]:
                txt.write(u"$$")
                txt.write(str(anal[2]))
            txt.write(u"||")
        txt.write(u"}}")
        out = txt.getvalue()[:-4]+u"}}"
    else:
        out = txt.getvalue()
    return out
    
def parse_text(txt):
    lines = txt.strip().split(u"\n")
    return [ [ (txt2triplet(tok)) for tok in line.split()] for line in lines]

def parse_scoredsent(txt):
    s_pos = txt.find(u"$$")
    score = float(txt[s_pos+2:-2])
    rest = txt[:s_pos]
    tokens = rest.split(u" ")
    tokens = list(map(txt2triplet, tokens))
    return (tokens, score)