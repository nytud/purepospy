#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

import os
import sys
import math
from itertools import chain
from json import loads as json_loads

# For PurePOS TCP server
import socketserver
from datetime import datetime

import jnius_config


def import_pyjnius():
    """
    PyJNIus can only be imported once per Python interpreter and one must set the classpath before importing...
    """
    # Check if autoclass is already imported...
    if not jnius_config.vm_running:

        # Tested on Ubuntu 16.04 64bit with openjdk-8 JDK and JRE installed:
        # sudo apt install openjdk-8-jdk-headless openjdk-8-jre-headless

        # Set JAVA_HOME for this session
        try:
            os.environ['JAVA_HOME']
        except KeyError:
            os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-8-openjdk-amd64/'

        # Set path and import jnius for this session
        from jnius import autoclass
    else:
        import sys
        from jnius import cast, autoclass  # Dummy autoclass import to silence the IDE
        class_loader = autoclass('java.lang.ClassLoader')
        cl = class_loader.getSystemClassLoader()
        ucl = cast('java.net.URLClassLoader', cl)
        urls = ucl.getURLs()
        cp = ':'.join(url.getFile() for url in urls)

        print('Warning: PyJNIus is already imported with the following classpath: {0} Please check if it is ok!'.
              format(cp), file=sys.stderr)

    # Return autoclass for later use...
    return autoclass


class UserMorphology:
    def __init__(self, anals):
        self.anals = anals

    def stem(self, pos, _):
        return self.anals[pos]


class PurePOS:
    class_path = ':'.join((os.path.join(os.path.dirname(__file__),
                                        'purepos/purepos-2.1-dev.one-jar/', jar)
                           for jar in ('lib/guava-r09.jar',
                                       'lib/commons-lang3-3.0.1.jar',
                                       'main/purepos-2.1-dev.jar')))

    def __init__(self, model_name=os.path.join(os.path.dirname(__file__), 'purepos/szeged.model'), morphology=None,
                 source_fields=None, target_fields=None):
        if not jnius_config.vm_running:
            jnius_config.add_classpath(PurePOS.class_path)
            self._autoclass = import_pyjnius()
        self._params = {}
        self._model_name = model_name

        # We have to use it later...
        self._java_string_class = self._autoclass('java.lang.String')
        self._java_list_class = self._autoclass('java.util.ArrayList')
        self._java_pair_class = self._autoclass('org.apache.commons.lang3.tuple.Pair')
        self._java_anal_item_class = self._autoclass('hu.ppke.itk.nlpg.purepos.common.TAnalysisItem')

        self._model_jfile = self._java_string_class(self._model_name.encode('UTF-8'))
        self.morphology = morphology
        self._model = None
        self._tagger = None

        # Field names for e-magyar TSV
        if source_fields is None:
            source_fields = {}

        if target_fields is None:
            target_fields = []

        self.source_fields = source_fields
        self.target_fields = target_fields

    def train(self, sentences, tag_order=2, emission_order=2, suff_length=10, rare_freq=10,
              lemma_transformation_type='suffix', lemma_threshold=2):
        self._params.update({'tag_order': tag_order, 'emission_order': 2, 'suff_length': 10, 'rare_freq': 10,
                             'lemma_transformation_type': 'suffix', 'lemma_threshold': 2})

        # 1) Load Serializer
        serializer = self._autoclass('hu.ppke.itk.nlpg.purepos.common.serializer.SSerializer')()

        # 2) Check if the modell file exists yes-> append new training data to model, no-> create new model
        if os.path.exists(self._model_name):
            print('Reading previous model...', end='', file=sys.stderr)
            self._model = serializer.readModelEx(self._model_jfile)
            print('Done', file=sys.stderr)
        else:
            self._model = self._autoclass('hu.ppke.itk.nlpg.purepos.model.internal.RawModel')(tag_order, emission_order,
                                                                                              suff_length, rare_freq)
        # 3) Set lemmatisation parameters to model
        self._model.setLemmaVariables(lemma_transformation_type, lemma_threshold)

        # 4) Read sentences
        # 4a) Load the required classses
        print('Reading sentences...', end='', file=sys.stderr)
        document_java_class = self._autoclass('hu.ppke.itk.nlpg.docmodel.internal.Document')
        paragraph_java_class = self._autoclass('hu.ppke.itk.nlpg.docmodel.internal.Paragraph')
        sentence_java_class = self._autoclass('hu.ppke.itk.nlpg.docmodel.internal.Sentence')
        token_java_class = self._autoclass('hu.ppke.itk.nlpg.docmodel.internal.Token')
        java_list_class = self._java_list_class

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
        serializer.writeModelEx(self._model, self._model_jfile)
        print('Done', file=sys.stderr)

    def tag_sentence(self, sent, beam_log_theta=math.log(1000), suffix_log_theta=math.log(10), max_guessed=10,
                     use_beam_search=False, lemma_transformation_type='suffix', lemma_threshold=2):
        if self._tagger is None:
            print('Compiling model...', end='', file=sys.stderr)
            # DUMMY STUFF NEEDED LATTER:
            # 1) Create nullanalyzer as we will use a Morphological Alalyzer form outside
            analyzer = self._autoclass('hu.ppke.itk.nlpg.purepos.morphology.NullAnalyzer')()
            # 2) Create dummy configuration
            conf = self._autoclass('hu.ppke.itk.nlpg.purepos.cli.configuration.Configuration')()
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
            serializer = self._autoclass('hu.ppke.itk.nlpg.purepos.common.serializer.SSerializer')
            # 3) Deserialize. Here we need the dependencies to be in the CLASS_PATH: Guava and lang3
            read_mod = serializer().readModelEx(self._model_jfile)
            # 4) Compile the model
            compiled_model = read_mod.compile(conf, lemma_transformation_type, lemma_threshold)

            # FIRE UP THE TAGGER WITH THE GIVEN ARGUMENTS:
            self._tagger = self._autoclass('hu.ppke.itk.nlpg.purepos.MorphTagger')(
                compiled_model, analyzer, beam_log_theta, suffix_log_theta, max_guessed, use_beam_search)
            print('Done', file=sys.stderr)

        # Here we add the Morphological Analyzer's analyses when there are...
        """
        MA(pos, tok) -> list(*(lemma, TAG))
        EmMorphPy ignores pos, the preanalysed input ignores tok argument...
        This input goes into the tagger
        """
        if self.morphology is not None:
            morph = self.morphology
        else:
            morph = self._dummy_morphology

        java_pair = self._java_pair_class
        jt_analysis_item = self._java_anal_item_class
        java_list_class = self._java_list_class

        new_sent = java_list_class()
        for pos, tok in enumerate(sent):
            # Create anals in native JAVA type format
            anals = java_list_class()
            for lemma, tag in morph(pos, tok):
                anals.add(jt_analysis_item.create(self._java_string_class(lemma.encode('UTF-8')),
                                                  self._java_string_class(tag.encode('UTF-8'))))

            # Create sentence in native JAVA type format
            new_sent.add(java_pair.of(self._java_string_class(tok.encode('UTF-8')), anals))

        ret = self._tagger.tagSentenceEx(new_sent)
        for i in range(ret.size()):
            t = ret.get(i)
            yield (t.token, t.stem, t.tag)

    @staticmethod
    def prepare_fields(field_names):
        # TODO: Maybe its not a good idea to hand-wire here the name and order of the features
        return [field_names['string'], field_names['anas']]

    def process_sentence(self, sen, field_indices):
        sent = []
        m = {}
        for pos, tok in enumerate(sen):
            token = tok[field_indices[0]]
            sent.append(token)
            # TODO: Maybe its not a good idea to hand-wire here the name and order of the features
            m[pos] = [(ana['lemma'], ana['tag']) for ana in json_loads(tok[field_indices[1]])]  # lemma, tag

        self.morphology = lambda position, _: UserMorphology(m).anals[position]
        for tok, (_, lemma, hfstana) in zip(sen, self.tag_sentence(sent)):
            tok.extend([lemma, hfstana])
        return sen

    @staticmethod
    def _dummy_morphology(*_):
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
        self._morphology = lambda pos, tok: EmMorphPy().stem(tok)
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
            tokens = self.tokenizer.word_tokenize(sent)
            ret = ['#'.join(tok) for tok in self.purepos.tag_sentence(tokens)]
            yield ret

    # No tokenisation. It waits for exactly one sentence in the OLD purepos format!
    def process_preanalyzed_text(self, input_text):
        anals = {}
        plain_sent = []
        for pos, tok in enumerate(input_text.split('\n', maxsplit=1)[0].split()):
            if tok.endswith('}}') and '{{' in tok:
                tok, rest = tok[:-2].split('{{', maxsplit=1)
                if len(rest) > 0:
                    for anal in rest.split('||'):
                        lemma, tag = anal.split('[', maxsplit=1)
                        tag = '[' + tag
                        anals[pos] = (lemma, tag)
            plain_sent.append(tok)
        self.purepos.morphology = UserMorphology(anals)
        # REST API waits for the first of the setences
        return [' '.join('#'.join(tok) for tok in self.purepos.tag_sentence(input_text))]

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
