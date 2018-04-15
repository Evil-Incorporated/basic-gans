#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
import json
from os.path import join

import subprocess
import urllib.request

__author__ = 'Fisher Yu'
__email__ = 'fy@cs.princeton.edu'
__license__ = 'MIT'

output_directory = './data' # Change this to the name of the directory where you want the dataset to be downloaded


def list_categories(tag):
    url = 'http://lsun.cs.princeton.edu/htbin/list.cgi?tag=' + tag
    f = urllib.request.urlopen(url)
    return json.loads(f.read().decode('utf-8'))


def download(out_dir, category, set_name, tag):
    url = 'http://lsun.cs.princeton.edu/htbin/download.cgi?tag={tag}' \
          '&category={category}&set={set_name}'.format(**locals())
    if set_name == 'test':
        out_name = 'test_lmdb.zip'
    else:
        out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
    out_path = join(out_dir, out_name)
    cmd = ['curl', url, '-o', out_path]
    print('Downloading', category, set_name, 'set')
    subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default='latest')
    parser.add_argument('-o', '--out_dir', default=output_directory)
    parser.add_argument('-c', '--category', default='church_outdoor')
    args = parser.parse_args()

    categories = list_categories(args.tag)
    if args.category is None:
        print('Downloading', len(categories), 'categories')
        for category in categories:
            download(args.out_dir, category, 'train', args.tag)
            download(args.out_dir, category, 'val', args.tag)
        download(args.out_dir, '', 'test', args.tag)
    else:
        if args.category == 'test':
            download(args.out_dir, '', 'test', args.tag)
        elif args.category not in categories:
            print('Error:', args.category, "doesn't exist in",
                  args.tag, 'LSUN release')
        else:
            download(args.out_dir, args.category, 'train', args.tag)
            download(args.out_dir, args.category, 'val', args.tag)


if __name__ == '__main__':
    main()
