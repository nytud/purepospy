#!/usr/bin/python3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

import os
import sys
import math
from json import loads as json_loads

from xtsv import jnius_config, import_pyjnius


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
    pass_header = True

    def __init__(self, model_name=os.path.join(os.path.dirname(__file__), 'purepos/szeged.model'), morphology=None,
                 source_fields=None, target_fields=None):
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
            source_fields = set()

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
        return [field_names['form'], field_names['anas']]

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


if not jnius_config.vm_running:
    jnius_config.add_classpath(PurePOS.class_path)
