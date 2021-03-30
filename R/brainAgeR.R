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
#' @param templateBrainMask input template brain mask, optional
#' @return preprocessing in a list
#' \itemize{
#'   \item{"imageAffine": }{Affine transformed and intensity normalized image.}
#'   \item{"brainMask": }{brain extraction probability mask.}
#'   \item{"brainMaskAffine": }{brain extraction probability mask, affine transformed.}
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
brainAgePreprocessing <- function( x, template, templateBrainMask ) {
  library( keras )
  if ( missing( template ) ) {
    templateFN = system.file("extdata", "template.nii.gz", package = "brainAgeR", mustWork = TRUE)
    templateFNB = system.file("extdata", "template_brain.nii.gz", package = "brainAgeR", mustWork = TRUE)
    template = antsImageRead( templateFN )
    templateBrainMask = antsImageRead( templateFNB )
    }
  tardim = c( 192, 224, 192 )
  template = resampleImage( template, tardim , useVoxels=TRUE, interpType = 'linear' )
  templateBrain = template * resampleImageToTarget( templateBrainMask, template )
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
      brainMaskAffine = bxtAff,
      biasField = biasField,
      affineMapping = aff ) )
}




#' getBrainAgeModel
#'
#' Create the brain age model data, downloading data as necessary. Data will be
#' downloaded from \url{https://figshare.com/articles/pretrained_networks_for_deep_learning_applications/7246985}
#'
#' @param modelPrefix prefix identifying directory for model file locations following \code{load_model_weights_tf} (optional)
#' @return tensorflow model
#' @author Avants BB
#' @examples
#'
#' \dontrun{
#' mdl = getBrainAgeModel( tempfile() )
#' }
#' @export
getBrainAgeModel <- function( modelPrefix ) {
  posts = c(
    "brainAge2020_att3.index",
    "brainAge2020_att3.data-00000-of-00002",
    "brainAge2020_att3.data-00001-of-00002"
  )
  if ( ! missing( modelPrefix ) ) {
    mdlfns = paste0( modelPrefix, "/", posts )
    myurl = "https://ndownloader.figshare.com/files/22365378"
    if ( ! file.exists( mdlfns[1] ) ) {
      tempfile = tempfile()
      download.file( myurl, tempfile )
      zip::unzip( tempfile, exdir =  modelPrefix )
      unlink( tempfile )
      }
    modelfn = paste0( modelPrefix, "/brainAge2020_att3"  )
    if ( ! all( file.exists( mdlfns )  ) )
      stop( paste("download fail: please download  from", myurl, "and place in directory", modelPrefix) )
  }
  nclass = 7
  ncogs = 1
  nChannels = 4
  #############
  efficientAttention <- function( inputX, nf=16L, pool_size=2L, kernel_size = 3,
    instanceNormalization = FALSE, targetDimensionality = 3,
    concatenate = FALSE, wt = 0.9 ) {
    outputType = 'basic'
    if ( wt > 1 ) wt = 1
    if ( wt < 0 ) wt = 0
  if ( outputType == "none" ) return( inputX )
  if ( ! outputType %in% c("basic","multiply","concatenate","attention") )
    stop( paste( "outputType", outputType, "not one of basic multiply concatenate or attention" ) )
  if ( targetDimensionality == 2 ) {
    myconv = layer_conv_2d
    mypool = layer_max_pooling_2d
    myup = layer_upsampling_2d
  } else {
    myconv = layer_conv_3d
    mypool = layer_max_pooling_3d
    myup = layer_upsampling_3d
  }
  getShape <- function( shapein, targetDimensionality ) {
    inshape = c(  )
    for ( k in 2:(2+targetDimensionality) ) inshape[k-1] = shapein$shape[[k]]
    return( c( NULL,
      as.integer( inshape[targetDimensionality+1]),
      as.integer(  prod( inshape[1:targetDimensionality] ) ) ) )
  }
  if ( instanceNormalization ) {
    f <- inputX %>%
      myconv( nf, kernel_size, padding='same'  ) %>%
        layer_instance_normalization(  )
    } else {
      f <- inputX %>%
        myconv( nf, kernel_size, padding='same'  ) # %>% layer_activation_selu()
    }
  f <- f %>% mypool(pool_size=rep(pool_size,targetDimensionality) )
  flatf = f %>% layer_reshape( getShape( f, targetDimensionality ) )
  s = tf$matmul( flatf, flatf, transpose_b = TRUE )
  beta = tf$nn$softmax( s )  # attention map
  g <- inputX %>%
    myconv( nf, kernel_size, padding='same'  ) # %>% layer_activation_selu()
  g <- g %>% mypool( pool_size = rep( pool_size, targetDimensionality ) )
  flatg = g %>% layer_reshape( getShape( g, targetDimensionality ) )
  o = tf$matmul( beta, flatg )  # [bs, N, C]
  targetShape1 = as.integer( (inputX$shape)[[2]] )
  targetShape2 = as.integer( (inputX$shape)[[3]] )
  reshapeVal1 = as.integer( targetShape1 / pool_size)
  reshapeVal2 = as.integer( targetShape2 / pool_size)
  lastChan = as.integer( o$shape[[2]] )
  nChannels = as.integer( (inputX$shape)[[4]] )
  if ( targetDimensionality == 3 ) {
    targetShape3 = as.integer( (inputX$shape)[[4]] )
    reshapeVal3 = as.integer( targetShape3 / pool_size)
    nChannels = as.integer( (inputX$shape)[[5]] )
    }
  if ( targetDimensionality == 3 ) {
    o = o %>% layer_reshape( c(NULL, reshapeVal1, reshapeVal2, reshapeVal3,  lastChan ) )
    }
  if ( targetDimensionality == 2 )
    o = o %>% layer_reshape( c(NULL, reshapeVal1, reshapeVal2,  lastChan ) )
  if ( pool_size > 1 )
    o = o %>% myup( pool_size )
  convo = myconv( o, nChannels, 1, activation='relu', padding='same'  )
  if ( outputType == "concatenate" ) {
    myatt = layer_concatenate( list( inputX, convo ) ) %>%
      layer_dense( nChannels )
  } else if ( outputType == "attention") {
    return( convo )
  } else if ( outputType == "multiply") {
    myatt = layer_multiply( list( inputX, convo ) )
  } else myatt = tf$math$multiply( inputX, tf$cast(wt,inputX$dtype) )  + 
     tf$math$multiply( convo , tf$cast(1.0 - wt,inputX$dtype) ) # sigma should be absorbed into conv values

  }

  ################################################################################
  myinput <- layer_input( list(96,112,96,4) )
  firstLayer <- efficientAttention( myinput, 8L, pool_size = 8L,
    instanceNormalization = FALSE,
    targetDimensionality = 3, concatenate = FALSE )
  mdl <- ANTsRNet::createResNetModel3D(
    list(NULL,NULL,NULL,4), numberOfClassificationLabels = 1,
         layers = 1:4, residualBlockSchedule = c(3, 4, 6, 3),
         lowestResolution = 64, cardinality = 64, mode = "regression")
  ################################################################################
  layerName = as.character(
    mdl$layers[[length(mdl$layers)-1 ]]$name )
  idLayer <- layer_dense( get_layer(mdl, layerName )$output, nclass,
    activation='softmax' )
  ageLayer <- layer_dense( get_layer(mdl, layerName )$output, 1, activation = 'linear' )
  sexLayer <- layer_dense( get_layer(mdl, layerName )$output, 1,
    activation = 'sigmoid' )
  mdlFull <- keras_model( inputs = mdl$input,
      outputs = list(
        idLayer,
        ageLayer,
        sexLayer ) )
  mdl2 = mdlFull( firstLayer )
  mdlFull <- keras_model(
    inputs = myinput,
      outputs = mdl2 )
  if ( ! missing( modelPrefix ) ) load_model_weights_tf( mdlFull, modelfn )
  mdlFull %>% compile(
    optimizer = optimizer_adam( lr = 1e-4 ),
    loss = list( "categorical_crossentropy", "mae", "binary_crossentropy" ),
    loss_weights = c( 1./9., 0.1, 1. ),
    metrics = list('accuracy') )
  return( mdlFull )
}


#' brainAge
#'
#' Estimate brain age and related variable from input T1 MRI.
#'
#' @param x input image
#' @param template input template, optional
#' @param templateBrainMask input template brain mask, optional
#' @param model input deep model, see \code{getBrainAgeModel}
#' @param polyOrder optional polynomial order for intensity matching (e.g. 1)
#' @param batch_size greater than 1 uses simulation to add variance in estimated values
#' @param sdAff larger values induce more variance
#' @return data frame of predictions and the brain age model
#' @author Avants BB
#' @examples
#'
#' \dontrun{
#' library( brainAgeR )
#' library( ANTsR )
#' library( keras )
#' filename = system.file("extdata", "test_image.nii.gz", package = "brainAgeR", mustWork = TRUE)
#' img = antsImageRead( filename ) # T1 image
#' mdl = getBrainAgeModel( tempfile() )
#' bage = brainAge( img, batch_size = 10, sdAff = 0.01, model = mdl )
#' bage[[1]][,1:4]
#' }
#'
#' @export brainAge
#' @importFrom stats rnorm
#' @importFrom ANTsRNet createResNetModel3D randomImageTransformAugmentation linMatchIntensity
#' @importFrom ANTsRCore antsRegistration antsApplyTransforms
brainAge <- function( x,
  template,
  templateBrainMask,
  model,
  polyOrder,
  batch_size = 8,
  sdAff = 0.01 ) {
  library( keras )
  if ( missing( template ) ) {
    templateFN = system.file("extdata", "template.nii.gz", package = "brainAgeR", mustWork = TRUE)
    templateFNB = system.file("extdata", "template_brain.nii.gz", package = "brainAgeR", mustWork = TRUE)
    template = antsImageRead( templateFN )
    templateBrainMask = antsImageRead( templateFNB )
    }
  tardim = c( 192, 224, 192 )
  template = resampleImage( template, tardim , useVoxels=TRUE, interpType = 'linear' )
  templateBrain = template * resampleImageToTarget( templateBrainMask, template )
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

  nChannels = 4

  imageAff = baprepro$imageAffine
  imageAffSub = resampleImageToTarget( imageAff, templateSub )
  if ( ! missing( "polyOrder" ) ) {
    imageAff = ANTsRNet::linMatchIntensity( imageAff, avgImg, polyOrder = polyOrder, truncate = TRUE )
    imageAffSub = ANTsRNet::linMatchIntensity( imageAffSub, avgImg2, polyOrder = polyOrder, truncate = TRUE )
    }
  fullDims = dim( imageAff )
  myAug3D <- function( fullImage, brainMask, batch_size = 1, sdAff = 0.0 ) {
        X = array( dim = c( batch_size, dim( templateSub ), nChannels ) )
        # X2 = array( dim = c( batch_size, dim( templateSub ), nc ) )
        bmask = thresholdImage( brainMask, 0.33, Inf )
        fullImage = brainAgeR::standardizeIntensity( fullImage, bmask ) * bmask
        randy = ANTsRNet::randomImageTransformAugmentation( fullImage,
          interpolator = c( "linear", "linear" ),
          list( list( fullImage ) ), list( fullImage ), sdAffine = sdAff, n = batch_size )
        for ( ind in 1:batch_size ) {
          fullImage = randy$outputPredictorList[[ind]][[1]]
          imgG = resampleImageToTarget( fullImage, avgImg2 )
          imgGdiff = imgG - avgImg2
          pdiff = fullImage - avgImg
          patch = cropIndices( fullImage, dim(fullImage)/4, dim(fullImage)/4+dim(fullImage)/2-1)
          pdiff = cropIndices( pdiff, dim(fullImage)/4, dim(fullImage)/4+dim(fullImage)/2-1)
          X[ ind, , , , 1 ] = as.array( imgG ) #  * 255 - 127.5
          X[ ind, , , , 2 ] = as.array( imgGdiff ) # * 255 - 127.5
          X[ ind, , , , 3 ] = as.array( patch ) #  * 255 - 127.5
          X[ ind, , , , 4 ] = as.array( pdiff ) # * 255 - 127.5
#          X2[ind, , , , 1 ] = as.array( patch ) #  * 255 - 127.5
#          X2[ind, , , , 2 ] = as.array( pdiff ) # * 255 - 127.5
        }
      return( X )
      }

  myX = myAug3D( imageAff, baprepro$brainMaskAffine, batch_size = batch_size, sdAff = sdAff )
  pp = predict( model, myX )
  sitenames = c( "ADNI", "DLBS","HCP","IXI","NKIRockland","OAS1_","SALD" )
  mydf = data.frame(
    predictedAge = as.numeric( pp[[2]] ),
    predictedGender = as.numeric( pp[[3]] ) )
  siteDF = data.frame( matrix( pp[[1]], ncol = length( sitenames ) ) )
  names( siteDF ) = sitenames
  for ( k in 1:nrow( siteDF ) ) siteDF[k,] = siteDF[k,]/sum(siteDF[k,] )
  mydf <- cbind( mydf, siteDF )
  mydf$brainVolume = bvol
  return( list( predictions=mydf, model=model, array=myX ) )
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
