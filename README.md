
# brainAgeR

brain age with deep learning.


minimal example

```
library( brainAgeR )
library( ANTsR )
library( keras )
filename = system.file("extdata", "template.nii.gz", package = "brainAgeR", mustWork = TRUE)
img = antsImageRead( filename ) # T1 image
bage = brainAge( img )
bage[[1]][,1:4]
# should be around 67 years
```
