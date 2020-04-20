#!/usr/bin/env pyhton3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

from xtsv import build_pipeline, parser_skeleton, jnius_config


def main():
    argparser = parser_skeleton(description='PurePOSPy - a Python wrapper for PurePOS POS-tagger')
    opts = argparser.parse_args()

    jnius_config.classpath_show_warning = opts.verbose  # Suppress warning.
    conll_comments = opts.conllu_comments

    # Set input and output iterators...
    if opts.input_text is not None:
        input_data = opts.input_text
    else:
        input_data = opts.input_stream
    output_iterator = opts.output_stream

    # Set the tagger name as in the tools dictionary
    used_tools = ['pos']
    presets = []

    # Init and run the module as it were in xtsv

    # The relevant part of config.py
    em_tag = ('purepospy.purepospy', 'PurePOS', 'emTag (PurePOS)', (),
              {'source_fields': {'form', 'anas'}, 'target_fields': ['lemma', 'xpostag']})

    tools = [(em_tag, ('pos', 'emTag'))]

    # Run the pipeline on input and write result to the output...
    output_iterator.writelines(build_pipeline(input_data, used_tools, tools, presets, conll_comments))

    # TODO this method is recommended when debugging the tool
    # Alternative: Run specific tool for input (still in emtsv format):
    # output_iterator.writelines(process(input_iterator, inited_tools[used_tools[0]]))

    # Alternative2: Run REST API debug server
    # app = pipeline_rest_api('TEST', inited_tools, presets,  False)
    # app.run()


if __name__ == '__main__':
    main()
