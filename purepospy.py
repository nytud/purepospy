#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

import os
import sys
import math
from itertools import chain

# For PurePOS TCP server
import socketserver
from datetime import datetime

# Tested on Ubuntu 16.04 64bit with openjdk-8 JDK and JRE installed:
# sudo apt install openjdk-8-jdk-headless openjdk-8-jre-headless

# Set JAVA_HOME for this session
try:
    os.environ['JAVA_HOME']
except KeyError:
    os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-8-openjdk-amd64/'

# Set path for PurePOS for this session
purepos_dir = os.path.join(os.path.dirname(__file__), 'purepos/purepos-2.1-dev.one-jar/')

os.environ['CLASSPATH'] = ':'.join([os.path.join(purepos_dir, 'lib/guava-r09.jar'),
                                    os.path.join(purepos_dir, 'lib/commons-lang3-3.0.1.jar'),
                                    os.path.join(purepos_dir, 'main/purepos-2.1-dev.jar')])


# Set path and import jnius for this session
from jnius import autoclass


class UserMorphology:
    def __init__(self, anals):
        self.anals = anals

    def stem(self, token):
        return self.anals[token]


class PurePOS:
    def __init__(self, model_name, morphology=None):
        self._params = {}
        self._model_name = model_name
        self._java_string_class = autoclass('java.lang.String')  # We have to use it later...
        self._model_jfile = autoclass('java.io.File')(self._java_string_class(self._model_name.encode('UTF-8'))
        self.morphology = morphology
        self._model = None
        self._tagger = None
        self.target_fields = ['lemma', 'hfstana']  # For eMagyar TSV format...

    def train(self, sentences, tag_order=2, emission_order=2, suff_length=10, rare_freq=10,
              lemma_transformation_type='suffix', lemma_threshold=2):
        self._params.update({'tag_order': tag_order, 'emission_order': 2, 'suff_length': 10, 'rare_freq': 10,
                             'lemma_transformation_type': 'suffix', 'lemma_threshold': 2})

        # 1) Load Serializer
        serializer = autoclass('hu.ppke.itk.nlpg.purepos.common.serializer.SSerializer')()

        # 2) Check if the modell file exists yes-> append new training data to model, no-> create new model
        if os.path.exists(self._model_name):
            print('Reading previous model...', end='', file=sys.stderr)
            self._model = serializer.readModel(self._model_jfile)
            print('Done', file=sys.stderr)
        else:
            self._model = autoclass('hu.ppke.itk.nlpg.purepos.model.internal.RawModel')(tag_order, emission_order,
                                                                                        suff_length, rare_freq)
        # 3) Set lemmatisation parameters to model
        self._model.setLemmaVariables(lemma_transformation_type, lemma_threshold)

        # 4) Read sentences
        # 4a) Load the required classses
        print('Reading sentences...', end='', file=sys.stderr)
        document_java_class = autoclass('hu.ppke.itk.nlpg.docmodel.internal.Document')
        paragraph_java_class = autoclass('hu.ppke.itk.nlpg.docmodel.internal.Paragraph')
        sentence_java_class = autoclass('hu.ppke.itk.nlpg.docmodel.internal.Sentence')
        token_java_class = autoclass('hu.ppke.itk.nlpg.docmodel.internal.Token')
        java_list_class = autoclass('java.util.ArrayList')

        # 4b) Convert Python iterable to JAVA List and build a Document class
        sents = java_list_class()
        for sent in sentences:
            curr_sent = java_list_class()
            for tok, stem, tag in sent:
                curr_tok = token_java_class(self._java_string_class(tok.encode('UTF-8')),
                                            self._java_string_class(stem.encode('UTF-8')),
                                            self._java_string_class(tag.encode('UTF-8')))
                curr_sent.add(curr_tok)
            sents.add(sentence_java_class(curr_sent))

        paras = java_list_class()
        paras.add(paragraph_java_class(sents))
        doc = document_java_class(paras)

        print('Done', file=sys.stderr)
        # 5) Feed Document to the model to train
        print('Start training...', end='', file=sys.stderr)
        self._model.train(doc)
        print('Done', file=sys.stderr)
        # 6) Serializie model to file
        print('Writing model...', end='', file=sys.stderr)
        serializer.writeModel(self._model, self._model_jfile)
        print('Done', file=sys.stderr)

    def tag_sentence(self, sent, beam_log_theta=math.log(1000), suffix_log_theta=math.log(10), max_guessed=10,
                     use_beam_search=False, lemma_transformation_type='suffix', lemma_threshold=2):
        if self._tagger is None:
            print('Compiling model...', end='', file=sys.stderr)
            # DUMMY STUFF NEEDED LATTER:
            # 1) Create nullanalyzer as we will use a Morphological Alalyzer form outside
            analyzer = autoclass('hu.ppke.itk.nlpg.purepos.morphology.NullAnalyzer')()
            # 2) Create dummy configuration
            conf = autoclass('hu.ppke.itk.nlpg.purepos.cli.configuration.Configuration')()
            # 3) Create Log Theta for Beam
            self._params['beam_log_theta'] = beam_log_theta
            # 4) Create Log Theta for Suffix
            self._params['suffix_log_theta'] = suffix_log_theta
            # 5) Set Maximal number of guessed tags
            self._params['max_guessed'] = 10
            # 6) Do not use Beam Search, but use the other variant implemented instead
            self._params['use_beam_search'] = False

            # GET THE MODEL WORKING:
            # 1) Read model to JAVA File
            # Done in __init__()
            # 2) Create Serializer for deserializing
            serializer = autoclass('hu.ppke.itk.nlpg.purepos.common.serializer.SSerializer')
            # 3) Deserialize. Here we need the dependencies to be in the CLASS_PATH: Guava and lang3
            read_mod = serializer().readModel(self._model_jfile)
            # 4) Compile the model
            compiled_model = read_mod.compile(conf, lemma_transformation_type, lemma_threshold)

            # FIRE UP THE TAGGER WITH THE GIVEN ARGUMENTS:
            self._tagger = autoclass('hu.ppke.itk.nlpg.purepos.MorphTagger')(
                compiled_model, analyzer, beam_log_theta, suffix_log_theta, max_guessed, use_beam_search)
            print('Done', file=sys.stderr)

        # Here we add the Morphological Analyzer's analyses if there aren't any...
        """
        MA.stem(word) -> list(list(lemma, TAG)*)
        Is converted to lemmaTAG per anal.
        Which is  joined to anal1||anal2...
        Which is formated to word{{anal1||anal2}}
        Which is joined as input tokens if there is anal and token not ends with {{}}
        This input goes into the tagger
        And the string output's last character is stripped as there is an extra whitespace.
        """
        if self.morphology is not None:
            stem = self.morphology.stem
        else:
            stem = self._dummy_morphology

        new_sent = []
        for word in sent.split():
            if word.find('{{') == -1 and not word.endswith('}}'):
                word = '{0}{{{{{1}}}}}'.format(word, ('||'.join(''.join(anal) for anal in stem(word))))
            new_sent.append(word if not word.endswith('{{}}') else word[:-4])

        new_sent = ' '.join(new_sent)
        ret = self._tagger.tagSentence(self._java_string_class(new_sent.encode('UTF-8')))
        return ret.toString()[:-1]

    @staticmethod
    def prepare_fields(field_names):
        return [field_names['string'], field_names['anas']]

    @staticmethod
    def _add_ana_if_any(anas):
        out_anas = []
        if len(anas) > 3:
            """
            ana=alom[/N]=alm+a[Poss.3Sg]=a+[Nom], feats=[/N][Poss.3Sg][Nom], lemma=alom, readable_ana=alom[/N]=alm 
            + a[Poss.3Sg] + [Nom]};
            ana=alma[/N]=alma+[Nom], feats=[/N][Nom], lemma=alma, readable_ana=alma[/N] + [Nom]}
            """
            for ana in anas.split('};{'):
                ana_dict = {}
                for it in ana[1:-1].split(', '):
                    k, v = it.split('=', maxsplit=1)
                    ana_dict[k] = v

                tag = ana_dict['feats']
                lemma = ana_dict['lemma']
                out_anas.append((lemma, tag))
        return out_anas

    def process_sentence(self, sen, field_indices):
        sent = []
        m = UserMorphology({})
        for tok in sen:
            token = tok[field_indices[0]]
            sent.append(token)
            m.anals[token] = self._add_ana_if_any(tok[field_indices[1]])

        self.morphology = m
        for tok, tagged in zip(sen, self.tag_sentence(' '.join(sent)).split()):
            _, lemma, hfstana = tagged.split('#')
            tok.extend([lemma, hfstana])
        return sen

    @staticmethod
    def _dummy_morphology(_):
        return ()


def put_data_together(sents, annots):
    return [[(s, a[0], a[1]) for s, a in zip(*sent)] for sent in zip(sents, annots)]


def read_data_w_annotation(data, sent_sep='\n', word_sep=' ', field_sep='#'):
    sents = []
    annots = []
    for inp_sent in data.split(sent_sep):
        sent = []
        annot = []
        for wlt in inp_sent.split(word_sep):
            word, lemma, pos = wlt.split(field_sep)
            sent.append(word)
            annot.append((lemma, pos))
        sents.append(sent)
        annots.append(annot)
    return sents, annots


class Tokenizer:
    def __init__(self):
        self.abbr = {'a.', 'ac.', 'a.C.', 'adj.', 'adm.', 'ag.', 'agit.', 'AkH.', 'alez.', 'alk.', 'ált.',
                     'altbgy.', 'ang.', 'Aö.', 'ápr.', 'arch.', 'ásv.', 'at.', 'aug.', 'b.', 'Be.', 'bek.',
                     'belker.', 'berend.', 'Bfok.', 'biz.', 'bizt.', 'Bk.', 'bo.', 'Bp.', 'br.', 'bt.', 'Btét.',
                     'Btk.', 'Btke.', 'B.ú.é.k.', 'c.', 'Cal.', 'cc.', 'cca.', 'cf.', 'cif.', 'Co.', 'Colo.',
                     'Comp.', 'Copr.', 'Ctv.', 'cs.', 'Cs.', 'Csop.', 'cső.', 'csüt.', 'D.', 'dbj.', 'dd.',
                     'ddr.', 'de.', 'dec.', 'dikt.', 'dipl.', 'dk.', 'dny.', 'dolg.', 'dr.', 'Dr.', 'DR.', 'Dsz.',
                     'du.', 'Dzs.', 'é.', 'ea.', 'ed.', 'eff.', 'egyh.', 'ék.', 'ell.', 'elv.', 'elvt.', 'em.',
                     'eng.', 'eny.', 'ény.', 'érk.', 'Ész.', 'et.', 'etc.', 'eü.', 'ev.', 'évf.', 'ezr.', 'f.',
                     'fam.', 'f.é.', 'febr.', 'felügy.', 'felv.', 'ff.', 'ffi.', 'f.h.', 'fhdgy.',
                     'fil.', 'fiz.', 'Fla.', 'fm.', 'foglalk.', 'ford.', 'főig.', 'főisk.', 'Főszerk.', 'főtörm.',
                     'főv.', 'fp.', 'fr.', 'frsz.', 'fszla.', 'fszt.', 'ft.', 'fuv.', 'gazd.', 'gimn.', 'gk.', 'gkv.',
                     'GM.', 'gondn.', 'gör.', 'gr.', 'grav.', 'gy.', 'Gy.', 'gyak.', 'gyártm.', 'h.',
                     'hads.', 'hallg.', 'hdm.', 'hdp.', 'hds.', 'hg.', 'hiv.', 'hk.', 'HKsz.', 'hm.', 'Hmvh.', 'ho.',
                     'honv.', 'hőm.', 'hp.', 'hr.', 'hrsz.', 'hsz.', 'ht.', 'htb.', 'hv.', 'iá.', 'id.', 'i.e.', 'ifj.',
                     'ig.', 'igh.', 'ill.', 'imp.', 'ind.', 'inic.', 'int.', 'io.', 'ip.', 'ir.', 'irod.', 'isk.',
                     'ism.', 'i.sz.', 'izr.', 'j.', 'jan.', 'jav.', 'jegyz.', 'jjv.', 'jkv.', 'jogh.', 'jogt.',
                     'jr.', 'júl.', 'jún.', 'jvb.', 'k.', 'karb.', 'kat.', 'kb.', 'kcs.', 'kd.', 'képv.', 'ker.',
                     'kf.', 'kft.', 'kht.', 'kir.', 'kirend.', 'kísérl.', 'kisip.', 'kiv.', 'kk.', 'kkt.',
                     'klin.', 'K.m.f.', 'Kong.', 'Korm.', 'kóth.', 'könyvt.', 'körz.', 'köv.', 'közj.', 'közl.',
                     'közp.', 'közt.', 'kp.', 'Kr.', 'Kr.e.', 'krt.', 'Kr.u.', 'kt.', 'ktsg.', 'kult.', 'kü.',
                     'kv.', 'kve.', 'l.', 'lat.', 'ld.', 'legs.', 'lg.', 'lgv.', 'loc.', 'lt.', 'ltp.', 'luth.',
                     'm.', 'má.', 'márc.', 'Mass.', 'mat.', 'mb.', 'mé.', 'med.', 'megh.', 'mélt.', 'met.', 'mf.',
                     'mfszt.', 'miss.', 'mjr.', 'mjv.', 'mk.', 'Mlle.', 'Mme.', 'mn.', 'Mo.', 'mozg.', 'Mr.',
                     'Mrs.', 'Ms.', 'Mt.', 'mü.', 'műh.', 'műsz.', 'műv.', 'művez.', 'n.', 'nagyker.', 'nagys.',
                     'NB.', 'NBr.', 'neg.', 'nk.', 'N.N.', 'nov.', 'Nr.', 'nu.', 'ny.', 'Ny.', 'Nyh.', 'nyilv.',
                     'Nyr.', 'nyug.', 'o.', 'obj.', 'okl.', 'okt.', 'olv.', 'Op.', 'orsz.', 'ort.', 'ov.', 'ovh.',
                     'őrgy.', 'őrpk.', 'őrv.', 'össz.', 'ötk.', 'özv.', 'p.', 'pf.', 'pg.', 'P.H.', 'pk.', 'pl.',
                     'plb.', 'pld.', 'plur.', 'pol.', 'polg.', 'poz.', 'pp.', 'Pp.', 'prof.', 'Prof.', 'PROF.',
                     'prot.', 'P.S.', 'pság.', 'Ptk.', 'pu.', 'pü.', 'r.', 'rac.', 'rad.', 'red.', 'ref.', 'reg.',
                     'rev.', 'rf.', 'r.k.', 'rkp.', 'rkt.', 'röv.', 'rt.', 'rtg.', 'sa.', 'Salg.', 'sel.', 'sgt.',
                     's.k.', 'sm.', 'st.', 'St.', 'stat.', 'strat.', 'sz.', 'Sz.', 'szakm.', 'szaksz.',
                     'szakszerv.', 'szd.', 'szds.', 'szept.', 'szerk.', 'szf.', 'Szfv.', 'szimf.', 'Szjt.',
                     'szkv.', 'szla.', 'szn.', 'szolg.', 'szöv.', 'Szt.', 'Sztv.', 'szubj.', 't.', 'tanm.', 'tb.',
                     'tbk.', 'tc.', 'techn.', 'tek.', 'tf.', 'tgk.', 'tip.', 'tisztv.', 'titks.', 'tk.', 'tkp.',
                     'tny.', 'törv.', 'tp.', 'tszf.', 'tszk.', 'tszkv.', 'tü.', 'tv.', 'tvr.', 'Ty.', 'Tyr.',
                     'u.', 'ua.', 'ui.', 'Ui.', 'Új-Z.', 'ÚjZ.', 'úm.', 'ún.', 'unit.', 'uo.', 'út.', 'uv.',
                     'üag.', 'üd.', 'üdv.', 'üe.', 'ümk.', 'ütk.', 'üv.', 'v.', 'vál.', 'vas.', 'vb.', 'Vcs.',
                     'vegy.', 'vh.', 'vhol.', 'Vhr.', 'vill.', 'vízv.', 'vizsg.', 'vk.', 'vkf.', 'vkny.', 'vm.',
                     'vol.', 'vö.', 'vs.', 'vsz.', 'vv.', 'X.Y.', 'Zs.'}

    def sent_tokenize(self, input_text):
        """
        Checks for sentence ending followed by a whitespace and a word with the first letter capitalized
        The setence ending word should not be an abbreviation from the given list.
        :param input_text: raw input text
        :return: list of sentences
        """
        splited = input_text.split()
        sents = []
        last = 0
        for i, (word1, word2) in enumerate(zip(splited, splited[1:])):
            if word1.endswith(('.', '!', '?', '?!')) and word1 and word2[0].isupper() and word1 not in self.abbr:
                sents.append(splited[last:i+1])
                last = i+1
        sents.append(splited[last:])
        return sents

    def word_tokenize(self, inp_words):
        words = []
        for word in inp_words:
            if word not in self.abbr:
                if word.startswith(('\'', '"', '„', '»')):
                    words.append(word[0])
                    words.append('<g/>')  # Glue for detokenization...
                    word = word[1:]
                if word.endswith(('\'', '"', '”', '«', ',', '.', '?', '!', '?!')):
                    words.append(word[:-1])
                    words.append('<g/>')  # Glue for detokenization...
                    words.append(word[-1])
                else:
                    words.append(word)
            else:
                words.append(word)
        return words


class RawToPurePOS:
    def __init__(self):
        # FIRE UP emMorphPy
        from emmorphpy.emmorphpy import EmMorphPy
        print('Firing up emMorphPy...')
        self._morphology = EmMorphPy()
        # FIRE UP PurePOS for tagging
        print('Firing up PurePOSTagger...')
        self.purepos = PurePOS(model_name=os.path.join(os.path.dirname(__file__), 'szeged.model'),
                               morphology=self._morphology)
        # FIRE UP Tokenizer
        print('Firing up Tokenizer...')
        self.tokenizer = Tokenizer()

    def process_raw_text(self, input_text):
        out_text = list(chain.from_iterable(self.process_to_sentence(input_text)))  # Concatenate sentences
        return [out_text]  # REST API waits for the first of the setences

    def process_to_sentence(self, input_text):
        for sent in self.tokenizer.sent_tokenize(input_text):
            tokens = ' '.join(self.tokenizer.word_tokenize(sent))
            ret = self.purepos.tag_sentence(tokens)
            yield ret.split()

    # No tokenisation. It waits for exactly one sentence in the OLD purepos format!
    def process_preanalyzed_text(self, input_text):
        return [self.purepos.tag_sentence(input_text)]  # REST API waits for the first of the setences

    def close(self):
        pass


# https://docs.python.org/3/library/socketserver.html#socketserver-tcpserver-example
class PurePOSTCPHandler(socketserver.StreamRequestHandler):
    def handle(self):
        # while 1:
            # self.rfile is a file-like object created by the handler;
            # we can now use e.g. readline() instead of raw recv() calls
            data = self.rfile.readline()  # THIS CALL CAN NOT BLOCK!
            if len(data) > 0:
                data = data.decode('UTF-8').strip()
                # http://stackoverflow.com/a/5877368
                print('{0} -- {1} wrote: {2}'.format(str(datetime.now()), self.client_address[0], data))
                ret = repr(list(self.server.preprocess.process_raw_text(data)))
                print('{0} -- anwser to {1}: {2}'.format(str(datetime.now()), self.client_address[0], ret))
                # Likewise, self.wfile is a file-like object used to write back to the client
                self.wfile.write((ret+'\n').encode('UTF-8'))
                self.wfile.flush()

    @staticmethod
    def start_server(host='localhost', port=9999):
        # Create the server, binding to localhost on port 9999
        tcp_server = socketserver.TCPServer((host, port), PurePOSTCPHandler)
        tcp_server.preprocess = RawToPurePOS()
        print('Start serving on: {0}:{1}'.format(host, port))
        # Activate the server; this will keep running until you
        # interrupt the program with Ctrl-C
        tcp_server.serve_forever()


def test():

    # test_text = 'PurePOS-t betöltve, de "Javaból" elemzek.' \
    #             ' És ráadásul dr. Novák Attila "Pythonnal wrappelt" emMorphjával. Méghozzá több mondatot is.'

    test_text = 'A hétvégén a korábbi elnökjelölttel egyeztet a megválasztott elnök. ' \
                'Trump a sajtóhírek szerint miniszteri posztot ajánl fel a kampánya alatt őt kritizáló' \
                ' politikusnak.'
    preprocess = RawToPurePOS().process_raw_text(test_text)

    for sentence in preprocess:
        print(sentence)


if __name__ == '__main__':
    """
    sents = []
    with open('train.txt', encoding='UTF-8') as fh:
        for line in fh:
            line = line.strip()
            sent = []
            for tok in line.split(' '):
                sent.append(tok.split('#'))
            sents.append(sent)
    p = PurePOS('szeg.model')
    # p.train(sents)
    out = p.tag_sentence('Múlt év szeptemberében az osztállyal elmentünk kirándulni Balatonra .')
    print(out)
    exit(1)
    """
    test()
