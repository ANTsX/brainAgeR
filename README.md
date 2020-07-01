
# brainAgeR

brain age with deep learning.


minimal example

```
library( brainAgeR )
library( ANTsR )
library( tensorflow )
library( keras )
filename = system.file("extdata", "test_image.nii.gz", package = "brainAgeR", mustWork = TRUE)
img = antsImageRead( filename ) # T1 image
mdl = getBrainAgeModel( tempfile() )
bage = brainAge( img, batch_size = 10, sdAff = 0.01, model = mdl )
bage[[1]][,1:4]
# should be around 67 years
```
