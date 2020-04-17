#' standardizeIntensity
#'
#' Robust intensity standardization method
#'
#' @param x input image
#' @param mask input image mask
#' @param quantiles two-vector defining quantiles in the range of 0 to 1
#' @return image
#' @author Avants BB
#' @examples
#'
#' \dontrun{
#' imgn = standardizeIntensity( img )
#' }
#' @export
standardizeIntensity <- function( x, mask, quantiles = c(0.01,0.99) ) {
  if ( missing( mask ) ) mask = getMask( x )
  temp = x - quantile( x[mask==1], quantiles[1] )
  temp = temp / quantile( temp[mask==1], quantiles[2] )
  return( temp )
}

#' brainAgePreprocessing
#'
#' MRI preprocessing for brain age
#'
#' @param x input image
#' @param template input template, optional
#' @return preprocessing in a list
#' \itemize{
#'   \item{"imageAffine": }{Affine transformed and intensity normalized image.}
#'   \item{"brainMask": }{brain extraction probability mask.}
#'   \item{"biasField": }{\code{n4} bias field output}
#'   \item{"affineMapping": }{\code{antsRegistration} output}
#'   }
#' @author Avants BB
#' @examples
#'
#' \dontrun{
#' myPre = brainAgePreprocessing( img )
#' }
#' @export
brainAgePreprocessing <- function( x, template ) {
  library( keras )
  if ( missing( template ) ) {
    templateFN = system.file("extdata", "template.nii.gz", package = "brainAgeR", mustWork = TRUE)
    templateFNB = system.file("extdata", "template_brain.nii.gz", package = "brainAgeR", mustWork = TRUE)
    }
  tardim = c( 192, 224, 192 )
  template = antsImageRead( templateFN )
  template = resampleImage( template, tardim , useVoxels=TRUE, interpType = 'linear' )
  templateBrain = template * resampleImageToTarget( antsImageRead( templateFNB ), template )
  templateSub = resampleImage( template, dim(template)/2,
            useVoxels=TRUE, interpType = 'linear' )
  avgimgfn1 = system.file("extdata", "avgImg.nii.gz", package = "brainAgeR", mustWork = TRUE)
  avgimgfn2 = system.file("extdata", "avgImg2.nii.gz", package = "brainAgeR", mustWork = TRUE)
  avgImg = antsImageRead( avgimgfn1 ) %>% antsCopyImageInfo2( template )
  avgImg2 = antsImageRead( avgimgfn2 ) %>% antsCopyImageInfo2( templateSub )
  meanMask = thresholdImage( x, 0.5 * mean( x ), Inf ) %>%
    morphology( "dilate", 3 ) %>% iMath("FillHoles")
  biasField = n4BiasFieldCorrection( x, meanMask, returnBiasField = T, shrinkFactor = 4 )
  bxt = brainExtraction( x / biasField )
  bxtThresh = thresholdImage( bxt, 0.5, Inf )
  biasField = n4BiasFieldCorrection( x, bxtThresh, returnBiasField = T, shrinkFactor = 4 )
  x = x / biasField
  bvol = prod( antsGetSpacing( bxt ) ) * sum( bxt )
  xBrain = x * bxtThresh
  aff = antsRegistration( iMath(templateBrain,"Normalize"), iMath( xBrain, "Normalize" ),
    "Affine", verbose = F )
  imageAff = antsApplyTransforms( template, x, aff$fwdtransforms,
        interpolator = c("linear") ) %>% iMath("Normalize")
  bxtAff = antsApplyTransforms( template, bxtThresh, aff$fwdtransforms,
              interpolator = c("nearestNeighbor") )
  imageAff = standardizeIntensity( imageAff, bxtAff, quantiles=c(0.01,0.99) )
  return(
    list(
      imageAffine = imageAff,
      brainMask = bxt,
      biasField = biasField,
      affineMapping = aff ) )
}

#' brainAge
#'
#' Estimate brain age and related variable from input T1 MRI
#'
#' @param x input image
#' @param template input template, optional
#' @param model input deep model, optional
#' @param polyOrder optional polynomial order for intensity matching (e.g. 1)
#' @param batch_size greater than 1 uses simulation to add variance in estimated values
#' @param sdAff larger values induce more variance
#' @return data frame of predictions and the brain age model
#' @author Avants BB
#' @examples
#'
#' \dontrun{
#' myPredictions = brainAge( img, template, model )
#' }
#' @export brainAge
#' @importFrom stats rnorm
#' @importFrom ANTsRNet createResNetModel3D randomImageTransformAugmentation linMatchIntensity
#' @importFrom ANTsRCore antsRegistration antsApplyTransforms
brainAge <- function( x, template, model, polyOrder, batch_size = 8,
  sdAff = 0.01 ) {
  library( keras )
  if ( missing( template ) ) {
    templateFN = system.file("extdata", "template.nii.gz", package = "brainAgeR", mustWork = TRUE)
    templateFNB = system.file("extdata", "template_brain.nii.gz", package = "brainAgeR", mustWork = TRUE)
    }
  tardim = c( 192, 224, 192 )
  template = antsImageRead( templateFN )
  template = resampleImage( template, tardim , useVoxels=TRUE, interpType = 'linear' )
  templateBrain = template * resampleImageToTarget( antsImageRead( templateFNB ), template )
  templateSub = resampleImage( template, dim(template)/2,
            useVoxels=TRUE, interpType = 'linear' )
  avgimgfn1 = system.file("extdata", "avgImg.nii.gz", package = "brainAgeR", mustWork = TRUE)
  avgimgfn2 = system.file("extdata", "avgImg2.nii.gz", package = "brainAgeR", mustWork = TRUE)
  avgImg = antsImageRead( avgimgfn1 ) %>% antsCopyImageInfo2( template )
  avgImg2 = antsImageRead( avgimgfn2 ) %>% antsCopyImageInfo2( templateSub )
  baprepro = brainAgePreprocessing( x )
  bxt = baprepro$brainMask
  bxtThresh = thresholdImage( bxt, 0.5, Inf )
  bvol = prod( antsGetSpacing( bxt ) ) * sum( bxt )
  getRandomBaseInd <- function( off = 10, patchWidth = 96 ) {
    baseInd = rep( NA, 3 )
    for ( k in 1:3 )
      baseInd[k]=sample( off:( fullDims[k] - patchWidth - off ) )[1]
    return( baseInd )
    }

    if ( missing( model ) ) {
      nclass = 6
      ncogs = 1
      modelFN = system.file("extdata", "resNet4LayerLR64Card64b.h5", package = "brainAgeR", mustWork = TRUE)
      inputImageSize = c( dim( templateSub ),  2  )
      mdl <- ANTsRNet::createResNetModel3D(inputImageSize, numberOfClassificationLabels = 1000,
             layers = 1:4, residualBlockSchedule = c(3, 4, 6, 3),
             lowestResolution = 64, cardinality = 64, mode = "classification")
      layerName = as.character(
        mdl$layers[[length(mdl$layers)-1 ]]$name )
      idLayer <- layer_dense( get_layer(mdl, layerName )$output, nclass,
        activation='sigmoid' ) # 'softmax' )
      ageLayer <- layer_dense( get_layer(mdl, layerName )$output, ncogs, activation = 'linear' )
      sexLayer <- layer_dense( get_layer(mdl, layerName )$output, 1,
        activation = 'sigmoid' )
      ptch = 96
      patchShape = c( rep( ptch, 3 ) , 2 )
      inputPatch <- layer_input( patchShape )
      model <- keras_model( inputs = list( mdl$input, inputPatch ),
          outputs = list(
            idLayer,
            ageLayer,
            sexLayer ) )
      load_model_weights_hdf5( model, modelFN )
      }

  imageAff = baprepro$imageAffine
  imageAffSub = resampleImageToTarget( imageAff, templateSub )
  if ( ! missing( "polyOrder" ) ) {
    imageAff = ANTsRNet::linMatchIntensity( imageAff, avgImg, polyOrder = polyOrder, truncate = TRUE )
    imageAffSub = ANTsRNet::linMatchIntensity( imageAffSub, avgImg2, polyOrder = polyOrder, truncate = TRUE )
    }
  fullDims = dim( imageAff )
  ptch = 96

  myAug3D <- function( img2, imgFull, batch_size = 1, sdAff = 0.0 ) {
        nc = 2
        X = array( dim = c( batch_size, dim( templateSub ), nc ) )
        X2 = array( dim = c( batch_size, rep(ptch,3), nc ) )
        for ( ind in 1:batch_size ) {
          imgG = img2
          if ( all(   dim(img2) == dim( avgImg2 ) ) ) {
            antsCopyImageInfo(avgImg2,  imgG )
            imgGdiff = imgG - avgImg2
          } else stop("dim(imgG) != dim( avgImg2 )")
          fullImage = imgFull
          if ( all(   dim(fullImage) == dim( avgImg ) ) )
            pdiff = fullImage - avgImg else stop("dim(fullImage) != dim( avgImg )")
          baseInd = getRandomBaseInd()
          topInd = baseInd + c( ptch, ptch, ptch ) - 1
          patch = cropIndices( fullImage, baseInd, topInd )
          pdiff = cropIndices( pdiff, baseInd, topInd )
          randy = ANTsRNet::randomImageTransformAugmentation( imgG,
            interpolator = c("linear","linear"),
            list( list( imgG, imgGdiff ) ), list( imgGdiff ), sdAffine = sdAff, n = 1 )
          imgG = randy$outputPredictorList[[1]][[1]]
          X[ ind, , , , 1 ] = as.array( imgG ) #  * 255 - 127.5
          X[ ind, , , , 2 ] = as.array( randy$outputPredictorList[[1]][[2]] ) # * 255 - 127.5
          X2[ind, , , , 1 ] = as.array( patch ) #  * 255 - 127.5
          X2[ind, , , , 2 ] = as.array( pdiff ) # * 255 - 127.5
        }
      return( list( X, X2 ) )
      }

  myX = myAug3D( imageAffSub, imageAff, batch_size = batch_size, sdAff = sdAff )
  pp = predict( model, myX )
  sitenames = c("DLBS","HCP","IXI","NKIRockland","OAS1_","SALD" )
  mydf = data.frame(
    predictedAge = as.numeric( pp[[2]] ),
    predictedGender = as.numeric( pp[[3]] ) )
  siteDF = data.frame( matrix( pp[[1]], ncol = length( sitenames ) ) )
  names( siteDF ) = sitenames
  for ( k in 1:nrow( siteDF ) ) siteDF[k,] = siteDF[k,]/sum(siteDF[k,] )
  mydf <- cbind( mydf, siteDF )
  mydf$brainVolume = bvol
  return( list( predictions=mydf, model=model ) )
}




#' brainExtraction
#'
#' ANTs brain extraction implemented with a u-net
#'
#' @param x input image
#' @param template input template, optional
#' @param model input deep model, optional
#' @param batch_size greater than 1 uses simulation to add variance in estimated values
#' @return brain extraction
#' @author Avants BB
#' @examples
#'
#' \dontrun{
#' myPredictions = brainExtraction( img )
#' }
#' @export brainExtraction
brainExtraction <- function( x, template, model, batch_size = 8 ) {
  #############################################################################################################
  bxtModelFN = system.file( "extdata", "bxtUnet.h5", package = "brainAgeR", mustWork = TRUE )
  templateFN = system.file( "extdata", "S_template3_resampled.nii.gz", package = "brainAgeR", mustWork = TRUE )
  reorientTemplate <- antsImageRead( templateFN )
  unetModel = load_model_hdf5( bxtModelFN )
  centerOfMassTemplate <- getCenterOfMass( reorientTemplate )
  centerOfMassImage <- getCenterOfMass(  x)
  xfrm <- createAntsrTransform( type = "Euler3DTransform",
      center = centerOfMassTemplate,
      translation = centerOfMassImage - centerOfMassTemplate )
  warpedImage <- applyAntsrTransformToImage( xfrm, x, reorientTemplate )
  resampledImageSize <- dim( reorientTemplate )
  batchX <- array( data = as.array( warpedImage ),
      dim = c( 1, resampledImageSize, 1 ) )
  batchX <- ( batchX - mean( batchX ) ) / sd( batchX )
  predictedData <- unetModel %>% predict( batchX, verbose = 0 )
  probabilityImage = as.antsImage( predictedData[1,,,,2] ) %>%
      antsCopyImageInfo2( reorientTemplate )
  probabilityImage <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
  probabilityImage,  x)
  return( probabilityImage )
  }
