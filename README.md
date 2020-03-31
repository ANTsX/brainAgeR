
# brainAgeR

brain age with deep learning.


minimal example

```
library( brainAgeR )
library( ANTsR )
library( keras )
filename = system.file("extdata", "test_image.nii.gz", package = "brainAgeR", mustWork = TRUE)
img = antsImageRead( filename ) # T1 image
bage = brainAge( img, batch_size = 10, sdAff = 0.1 )
bage[[1]][,1:4]
# should be around 67 years
```
