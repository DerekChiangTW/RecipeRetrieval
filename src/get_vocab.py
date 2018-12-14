#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import word2vec


'''
Usage: python get_vocab.py /path/to/vocab.bin
'''
w2v_file = sys.argv[1]
model = word2vec.load(w2v_file)
vocab = model.vocab

print "Writing to data/vocab.txt..."
output_dir = os.path.join('..', 'data')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, 'vocab.txt')
f = open(output_path, 'w')
f.write("\n".join(vocab))
f.close()
