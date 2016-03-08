# Artistic_Style
This is a matlab implementation of the paper [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.

## Setup
Install and compile [MatConvNet](http://www.vlfeat.org/matconvnet/)

Download [imagenet-vgg-verydeep-19](http://www.vlfeat.org/matconvnet/pretrained/) to /path/to/Artistic_Style.m

## Usage
Run Artistic_Style.m
Tested on OSX 10.11, Matlab 2014b, MatConvNet 1.0-beta16

## Details
Images are initialized with content image and optimized using L-BFGS.
