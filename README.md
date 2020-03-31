[![Build Status](https://travis-ci.org/ANTsX/ANTsRNet.png?branch=master)](https://travis-ci.org/ANTsX/ANTsRNet)

 <!-- badges: start -->
[![Build Status](https://travis-ci.com/muschellij2/ANTsRNet.png?branch=master)](https://travis-ci.com/muschellij2/ANTsRNet)
[![Codecov test coverage](https://codecov.io/gh/muschellij2/ANTsRNet/branch/master/graph/badge.svg)](https://codecov.io/gh/muschellij2/ANTsRNet?branch=master)
  <!-- badges: end -->

# brainAgeR

brain age with deep learning.


minimal example

```
library( brainAgeR )
library( ANTsR )
library( keras )
img = antsImageRead( filename ) # T1 image
bage = brainAge( img )
bage[[1]][,1:4]
```
