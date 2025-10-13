#include <Exceptions.h>
#include <util.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>
#include <cmath>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>

npp::ImageNPP_8u_C4 oArgyleTexture;
npp::ImageNPP_8u_C4 oRufflesTexture;

void loadImage(std::string sFilename, npp::ImageNPP_8u_C4 &rImage)
{
    int file_errors = 0;
    std::ifstream infile(sFilename.data(), std::ifstream::in);

    if (infile.good())
    {
        std::cout << "opened: <" << sFilename.data()
                  << "> successfully!" << std::endl;
        file_errors = 0;
        infile.close();
    }
    else
    {
        std::cout << "unable to open: <" << sFilename.data() << ">"
                  << std::endl;
        file_errors++;
        infile.close();
    }

    if (file_errors > 0)
    {
        exit(EXIT_FAILURE);
    }

    // declare a host image object for an 8-bit grayscale image
    npp::ImageCPU_8u_C4 oHost;
    // load gray-scale image from disk
    npp::loadImage(sFilename, oHost);
    // declare a device image and copy construct from the host image,
    // i.e. upload host to device
    npp::ImageNPP_8u_C4 oDevice(oHost);

    rImage.swap(oDevice);
}

NppiSize imageSizeROI(npp::ImageNPP_8u_C4 &oDeviceSrc)
{
    NppiSize oROI = {
        (int)oDeviceSrc.width(),
        (int)oDeviceSrc.height()};
    return oROI;
}

NppiSize imageSizeROI(npp::ImageNPP_8u_C1 &oDeviceSrc)
{
    NppiSize oROI = {
        (int)oDeviceSrc.width(),
        (int)oDeviceSrc.height()};
    return oROI;
}

NppiRect imageROI(npp::ImageNPP_8u_C4 &oDeviceSrc)
{
    NppiSize oSizeROI = imageSizeROI(oDeviceSrc);
    NppiRect oROI = {
        0,
        0,
        oSizeROI.width,
        oSizeROI.height};
    return oROI;
}

NppiRect moveROI(NppiRect oROI, NppiPoint to)
{
    NppiRect r = {
        oROI.x + to.x,
        oROI.y + to.y,
        oROI.width + to.x,
        oROI.height + to.y};
    return r;
}

void rotateT(npp::ImageNPP_8u_C4 &oDeviceSrc, double angle, NppiPoint shift, npp::ImageNPP_8u_C4 &oDeviceDst)
{
    // rotates oDeviceSrc by `angle` and translated by `shift`, reults in `oDeviceDSt`.

    NPP_CHECK_NPP(nppiRotate_8u_C4R(
        oDeviceSrc.data(), imageSizeROI(oDeviceSrc), oDeviceSrc.pitch(), imageROI(oDeviceSrc),
        oDeviceDst.data(), oDeviceDst.pitch(), imageROI(oDeviceDst),
        angle, -shift.x, -shift.y,
        // angle, 0, 0,
        NPPI_INTER_LINEAR));
}

void rotate(npp::ImageNPP_8u_C4 &oDeviceSrc, double angle, npp::ImageNPP_8u_C4 &oDeviceDst)
{
    // Calculate rotated bounding box.
    double aBoundingBox[2][2] = {0};
    NPP_CHECK_NPP(nppiGetRotateBound(imageROI(oDeviceSrc), aBoundingBox, angle, 0, 0));
    NppiRect oRotatedROI = {0,
                            0,
                            (int)(aBoundingBox[1][0] - aBoundingBox[0][0]),
                            (int)(aBoundingBox[1][1] - aBoundingBox[0][1])};

    npp::ImageNPP_8u_C4 oRotated(oRotatedROI.width, oRotatedROI.height);

    // shift to center rotated image.
    NppiPoint shift = {(int)aBoundingBox[0][0], (int)aBoundingBox[0][1]};
    rotateT(oDeviceSrc, angle, shift, oRotated);

    oDeviceDst.swap(oRotated);
}

void crop(npp::ImageNPP_8u_C4 &oDeviceSrc, NppiPoint oCropPoint, npp::ImageNPP_8u_C4 &oDeviceDst)
{
    // crops `oDeviceSrc` from `oCropPoint` to size of `oDeviceDst`, result in `oDeviceDst`.

    // There doesn't seem to be an easy crop function, so a 0 degree rotation is used instead.
    rotateT(oDeviceSrc, 0, oCropPoint, oDeviceDst);
}

void move(npp::ImageNPP_8u_C4 &oDeviceSrc, NppiPoint to, npp::ImageNPP_8u_C4 &oDeviceDst)
{
    // Moves `oDeviceSrc` to the point `to` in `oDeviceDst`.

    // A 0 degree rotation feels like the easiest way to do this.
    NPP_CHECK_NPP(nppiRotate_8u_C4R(
        oDeviceSrc.data(), imageSizeROI(oDeviceSrc), oDeviceSrc.pitch(), imageROI(oDeviceSrc),
        oDeviceDst.data(), oDeviceDst.pitch(), imageROI(oDeviceDst),
        0, to.x, to.y,
        NPPI_INTER_LINEAR));
}

void wrapTexture(npp::ImageNPP_8u_C4 &textureSrc, npp::ImageNPP_8u_C4 &oDeviceDSt)
{
    // Repeats the textureSrc over the oDeviceDst image, overwriting whatever is there.

    NppiSize oTextureSizeROI = imageSizeROI(textureSrc);
    NppiSize oDeviceSizeROI = imageSizeROI(oDeviceDSt);

    NPP_CHECK_NPP(nppiCopyWrapBorder_8u_C4R(
        textureSrc.data(), textureSrc.pitch(), oTextureSizeROI,
        oDeviceDSt.data(), oDeviceDSt.pitch(), oDeviceSizeROI,
        oDeviceSizeROI.height - oTextureSizeROI.height,
        oDeviceSizeROI.width - oTextureSizeROI.width));
}

void addTexture(npp::ImageNPP_8u_C4 &oDeviceSrc, npp::ImageNPP_8u_C4 &textureSrc, npp::ImageNPP_8u_C4 &oDeviceDSt)
{
    // Adds the textureSrc over the oDeviceSrc, outputs to the oDeivceDst.
    // textureSrc is repeated vertically and horizontally as needed to cover the oDeviceSrc.

    NppiSize oSrcSizeROI = imageSizeROI(oDeviceSrc);

    npp::ImageNPP_8u_C4 textureWrap(oDeviceDSt.width(), oDeviceDSt.height());
    wrapTexture(textureSrc, textureWrap);

    NPP_CHECK_NPP(nppiAdd_8u_C4RSfs(
        oDeviceSrc.data(), oDeviceSrc.pitch(),
        textureWrap.data(), textureWrap.pitch(),
        oDeviceDSt.data(), oDeviceDSt.pitch(),
        oSrcSizeROI, 0));
}

void addTextureROI(npp::ImageNPP_8u_C4 &oDeviceSrc, NppiRect oSrcROI, npp::ImageNPP_8u_C4 &textureSrc, npp::ImageNPP_8u_C4 &oDeviceDSt) {

    // Intentionally making a texture for the full destination region
    // and not just the ROI, so adding the same texture in multiple ROI is consistent.
    npp::ImageNPP_8u_C4 textureWrap(oDeviceDSt.width(), oDeviceDSt.height());
    wrapTexture(textureSrc, textureWrap);

    NppiSize oROI = {oSrcROI.width, oSrcROI.height};

    NPP_CHECK_NPP(nppiAdd_8u_C4RSfs(
        oDeviceSrc.data(oSrcROI.x, oSrcROI.y), oDeviceSrc.pitch(),
        textureWrap.data(), textureWrap.pitch(),
        oDeviceDSt.data(oSrcROI.x, oSrcROI.y), oDeviceDSt.pitch(),
        oROI, 0));
}

void addTextureTH(npp::ImageNPP_8u_C4 &oDeviceSrc, npp::ImageNPP_8u_C4 &textureSrc, npp::ImageNPP_8u_C4 &oDeviceDSt)
{
    // Adds `textureSrc` to only the top half of `oDeviceSrc`.

    NppiRect oSrcROI = imageROI(oDeviceSrc);
    oSrcROI.height /= 2;

    addTextureROI(oDeviceSrc, oSrcROI, textureSrc, oDeviceDSt);
}

void addTextureBH(npp::ImageNPP_8u_C4 &oDeviceSrc, npp::ImageNPP_8u_C4 &textureSrc, npp::ImageNPP_8u_C4 &oDeviceDSt)
{
    // Adds `textureSrc` to only the bottom half of `oDeviceSrc`.

    NppiRect oSrcROI = imageROI(oDeviceSrc);
    oSrcROI.height /= 2;
    oSrcROI.y += oSrcROI.height;

    addTextureROI(oDeviceSrc, oSrcROI, textureSrc, oDeviceDSt);
}

void addTextureMH(npp::ImageNPP_8u_C4 &oDeviceSrc, npp::ImageNPP_8u_C4 &textureSrc, npp::ImageNPP_8u_C4 &oDeviceDSt)
{
    // Adds `textureSrc` to only the Middle (between top and bottom quarters) half of `oDeviceSrc`.

    NppiRect oSrcROI = imageROI(oDeviceSrc);
    oSrcROI.height /= 2;
    oSrcROI.y += (oSrcROI.height / 2);

    addTextureROI(oDeviceSrc, oSrcROI, textureSrc, oDeviceDSt);
}

void addTextureTQ(npp::ImageNPP_8u_C4 &oDeviceSrc, npp::ImageNPP_8u_C4 &textureSrc, npp::ImageNPP_8u_C4 &oDeviceDSt)
{
    // Adds texture on only the top quarter of the image

    NppiRect oSrcROI = imageROI(oDeviceSrc);
    oSrcROI.height /= 4;

    addTextureROI(oDeviceSrc, oSrcROI, textureSrc, oDeviceDSt);
}

void addTextureMTQ(npp::ImageNPP_8u_C4 &oDeviceSrc, npp::ImageNPP_8u_C4 &textureSrc, npp::ImageNPP_8u_C4 &oDeviceDSt)
{
    // Adds texture on only the middle top quarter of the image

    NppiRect oSrcROI = imageROI(oDeviceSrc);
    oSrcROI.height /= 4;
    oSrcROI.y += oSrcROI.height;

    addTextureROI(oDeviceSrc, oSrcROI, textureSrc, oDeviceDSt);
}

void addTextureMBQ(npp::ImageNPP_8u_C4 &oDeviceSrc, npp::ImageNPP_8u_C4 &textureSrc, npp::ImageNPP_8u_C4 &oDeviceDSt)
{
    // Adds texture on only the middle bottom quarter of the image

    NppiRect oSrcROI = imageROI(oDeviceSrc);
    oSrcROI.height /= 4;
    oSrcROI.y += oSrcROI.height * 2;

    addTextureROI(oDeviceSrc, oSrcROI, textureSrc, oDeviceDSt);
}

void addTextureBQ(npp::ImageNPP_8u_C4 &oDeviceSrc, npp::ImageNPP_8u_C4 &textureSrc, npp::ImageNPP_8u_C4 &oDeviceDSt)
{
    // Adds texture on only the bottom quarter of the image

    NppiRect oSrcROI = imageROI(oDeviceSrc);
    oSrcROI.height /= 4;
    oSrcROI.y += oSrcROI.height * 3;

    addTextureROI(oDeviceSrc, oSrcROI, textureSrc, oDeviceDSt);
}

void rotateTexture(npp::ImageNPP_8u_C4 &textureSrc, double angle, npp::ImageNPP_8u_C4 &textureDst)
{
    // rotate `textureSrc` by `angle` such that textureDst titles correctly.
    // Doesn't tile exactly correctly, but its close enough for me.

    int width = textureSrc.width();
    int height = textureSrc.height();

    // Create tiled texture to crop the rotated version out of.
    npp::ImageNPP_8u_C4 textureWrap(width * 2, height * 2);
    wrapTexture(textureSrc, textureWrap);

    rotate(textureWrap, angle, textureWrap);

    NppiPoint oCropPoint = {width, height};
    npp::ImageNPP_8u_C4 oResult(width, height);
    crop(textureWrap, oCropPoint, oResult);

    textureDst.swap(oResult);
}

void makeGradient(npp::ImageNPP_8u_C4 &oDeviceGradient)
{
    // Creates a basic linear gradient from white to red from the top to bottom of oDeviceGradient.

    // Initial low resolution gradient.
    // Needs to be at least 2 across or resize gives a ROI error.
    npp::ImageNPP_8u_C4 oHostGradient(2, 4);

    NppiSize oROI = imageSizeROI(oHostGradient);
    NppiRect oRectROI = imageROI(oHostGradient);

    Npp8u white[] = {255, 255, 255, 255};
    Npp8u lightred[] = {191, 191, 255, 255};
    Npp8u darkred[] = {63, 63, 255, 255};
    Npp8u red[] = {0, 0, 255, 255};

    // Set image to red
    NPP_CHECK_NPP(nppiSet_8u_C4R(
        red,
        oHostGradient.data(), oHostGradient.pitch(),
        oROI));

    // Set everything above the bottom to dark red
    NppiSize otopROI = {oROI.width, 3};
    NPP_CHECK_NPP(nppiSet_8u_C4R(
        darkred,
        oHostGradient.data(), oHostGradient.pitch(),
        otopROI));

    // Set everything above the middle to light red
    otopROI = {oROI.width, 2};
    NPP_CHECK_NPP(nppiSet_8u_C4R(
        lightred,
        oHostGradient.data(), oHostGradient.pitch(),
        otopROI));

    // Make top edge white
    otopROI = {oROI.width, 1};
    NPP_CHECK_NPP(nppiSet_8u_C4R(
        white,
        oHostGradient.data(), oHostGradient.pitch(),
        otopROI));

    NppiSize oDeviceSizeROI = imageSizeROI(oDeviceGradient);
    NppiRect oDeviceRectROI = imageROI(oDeviceGradient);

    // Use resize linear interpolation to make a smooth gradient in oDeviceGradient.
    NPP_CHECK_NPP(nppiResize_8u_C4R(
        oHostGradient.data(), oHostGradient.pitch(), oROI, oRectROI,
        oDeviceGradient.data(), oDeviceGradient.pitch(), oDeviceSizeROI, oDeviceRectROI,
        NPPI_INTER_LINEAR));
}

const Npp8u *copyPallet(const Npp8u *pallet, int numElements)
{
    // Copy channel pallet array to device.
    size_t size = numElements * sizeof(Npp8u);
    Npp8u *d_pallet = NULL;
    NPP_CHECK_CUDA(cudaMalloc(&d_pallet, size));
    NPP_CHECK_CUDA(cudaMemcpy(d_pallet, pallet, size, cudaMemcpyHostToDevice));
    return d_pallet;
}

void downSampleA(npp::ImageNPP_8u_C4 &oDeviceSrc, const Npp8u *pTables[3], Npp8u bitDepth, npp::ImageNPP_8u_C4 &oDeviceDst)
{
    // Downsample `oDeviceSrc` using `pTables` in rgb channels to `bitDepth`, alpha is not downsampled.
    // pTables is a list of 3 host pointers for three channel pallets.
    // Channel order is blue, green, red.
    // Each channel must have `2^bitDepth` elements. TODO: add assert
    // BitDepth must be between (0,8] TODO: add assert

    int palletElements = pow(2, bitDepth);

    // copy pallet to device.
    const Npp8u *
        pallet[3] = {
            copyPallet(pTables[0], palletElements),
            copyPallet(pTables[1], palletElements),
            copyPallet(pTables[2], palletElements)};

    Npp8u shift = 8 - bitDepth;
    // Downsample to the 3 most significant bits.
    const Npp32u aConstants[4] = {shift, shift, shift, 0};
    NPP_CHECK_NPP(nppiRShiftC_8u_C4R(
        oDeviceSrc.data(), oDeviceSrc.pitch(), aConstants,
        oDeviceDst.data(), oDeviceDst.pitch(), imageSizeROI(oDeviceSrc)));

    // Recolor to pallet.
    NPP_CHECK_NPP(nppiLUTPalette_8u_AC4R(
        oDeviceDst.data(), oDeviceDst.pitch(),
        oDeviceDst.data(), oDeviceDst.pitch(),
        imageSizeROI(oDeviceSrc),
        pallet, bitDepth));
}

void downSampleA3(npp::ImageNPP_8u_C4 &oDeviceSrc, const Npp8u *pTables[3], npp::ImageNPP_8u_C4 &oDeviceDst)
{
    // Downsample `oDeviceSrc` using `pTables` to 8 values in rgb channels, alpha is not downsampled.
    // pTables is a list of 3 host pointers for three channel pallets.
    // Channel order is blue, green, red.
    // Each channel must have 8 elements.

    downSampleA(oDeviceSrc, pTables, 3, oDeviceDst);
}

void downSampleA2(npp::ImageNPP_8u_C4 &oDeviceSrc, const Npp8u *pTables[3], npp::ImageNPP_8u_C4 &oDeviceDst)
{
    // Downsample `oDeviceSrc` using `pTables` to 8 values in rgb channels, alpha is not downsampled.
    // pTables is a list of 3 host pointers for three channel pallets.
    // Channel order is blue, green, red.
    // Each channel must have 4 elements.

    downSampleA(oDeviceSrc, pTables, 2, oDeviceDst);
}

float contourCount(npp::ImageNPP_8u_C4 &oDeviceSrc)
{
    // Computes an image contour `oDeviceSrc` and returns what percent count as a contour.
    // Contours are calculated with nppiFilterCannyBorder_8u_C1R.

    NppiSize imageSize = imageSizeROI(oDeviceSrc);

    // Convert to grayscale for contour detection
    npp::ImageNPP_8u_C1 oDeviceGray(imageSize.width, imageSize.height);
    NPP_CHECK_NPP(nppiRGBToGray_8u_AC4C1R(
        oDeviceSrc.data(), oDeviceSrc.pitch(),
        oDeviceGray.data(), oDeviceGray.pitch(),
        imageSize));

    // Make buffer for contour calculation
    int nContourBufferSize = 0;
    Npp8u *pContourBufferNPP = 0;
    NPP_CHECK_NPP(nppiFilterCannyBorderGetBufferSize(imageSize, &nContourBufferSize));
    cudaMalloc((void **)&pContourBufferNPP, nContourBufferSize);

    Npp16s nLowThreshold = 200;
    Npp16s nHighThreshold = 500;
    npp::ImageNPP_8u_C1 oDeviceContour(imageSize.width, imageSize.height);
    NppiPoint oOrigin = {0, 0};
    NPP_CHECK_NPP(nppiFilterCannyBorder_8u_C1R(
        oDeviceGray.data(), oDeviceGray.pitch(), imageSize, oOrigin,
        oDeviceContour.data(), oDeviceContour.pitch(), imageSize,
        NPP_FILTER_SOBEL, NPP_MASK_SIZE_5_X_5,
        nLowThreshold, nHighThreshold,
        nppiNormL2, NPP_BORDER_REPLICATE,
        pContourBufferNPP));

    // Debug: copy contour to color image to save.
    // npp::ImageNPP_8u_C4 oDeviceDst(imageSize.width, imageSize.height);
    // NPP_CHECK_NPP(nppiCopy_8u_C1C4R(
    //     oDeviceContour.data(), oDeviceContour.pitch(),
    //     oDeviceDst.data(), oDeviceDst.pitch(),
    //     imageSize));
    // NPP_CHECK_NPP(nppiCopy_8u_C1C4R(
    //     oDeviceContour.data(), oDeviceContour.pitch(),
    //     oDeviceDst.data() + 1, oDeviceDst.pitch(),
    //     imageSize));
    // NPP_CHECK_NPP(nppiCopy_8u_C1C4R(
    //     oDeviceContour.data(), oDeviceContour.pitch(),
    //     oDeviceDst.data() + 2, oDeviceDst.pitch(),
    //     imageSize));

    // setup buffer for sum calculation.
    int sumBufferSize;
    Npp8u *sumDeviceBuffer;
    NPP_CHECK_NPP(nppiSumGetBufferHostSize_8u_C1R(imageSize, &sumBufferSize));
    cudaMalloc((void **)(&sumDeviceBuffer), sumBufferSize);

    // setup pointer for sum result.
    Npp64f *pSum;
    cudaMalloc((void **)(&pSum), sizeof(Npp64f));
    NPP_CHECK_NPP(nppiSum_8u_C1R(oDeviceContour.data(), oDeviceContour.pitch(), imageSizeROI(oDeviceContour), sumDeviceBuffer, pSum));

    // copy result to host.
    Npp64f nSumHost;
    cudaMemcpy(&nSumHost, pSum, sizeof(Npp64f), cudaMemcpyDeviceToHost);

    return (float)(nSumHost / 255.0) / (float)(imageSize.width * imageSize.height) * 100.0;
}

void blurROI(npp::ImageNPP_8u_C4 &oDeviceSrc, NppiRect oSrcROI, npp::ImageNPP_8u_C4 &oDeviceDst)
{
    NppiPoint origin = {0, 0};
    NppiSize oROI = {oSrcROI.width, oSrcROI.height};

    NPP_CHECK_NPP(nppiFilterGaussBorder_8u_C4R(
        oDeviceSrc.data(oSrcROI.x, oSrcROI.y), oDeviceSrc.pitch(), oROI, origin,
        oDeviceDst.data(oSrcROI.x, oSrcROI.y), oDeviceDst.pitch(), oROI,
        NPP_MASK_SIZE_5_X_5, NPP_BORDER_REPLICATE));
}

void sharpenROI(npp::ImageNPP_8u_C4 &oDeviceSrc, NppiRect oSrcROI, npp::ImageNPP_8u_C4 &oDeviceDst)
{
    NppiPoint origin = {0, 0};
    NppiSize oROI = {oSrcROI.width, oSrcROI.height};

    NPP_CHECK_NPP(nppiFilterSharpenBorder_8u_C4R(
        oDeviceSrc.data(oSrcROI.x, oSrcROI.y), oDeviceSrc.pitch(), oROI, origin,
        oDeviceDst.data(oSrcROI.x, oSrcROI.y), oDeviceDst.pitch(), oROI,
        NPP_BORDER_REPLICATE));
}

// TODO: make functions for half/quarter ROIs
// remove texture half/quarter functions
// make mutate, pick a function, then an roi
void dialateROI(npp::ImageNPP_8u_C4 &oDeviceSrc, NppiRect oSrcROI, npp::ImageNPP_8u_C4 &oDeviceDst)
{
    NppiPoint origin = {0, 0};
    NppiSize oROI = {oSrcROI.width, oSrcROI.height};

    NPP_CHECK_NPP(nppiDilate3x3Border_8u_C4R(
        oDeviceSrc.data(oSrcROI.x, oSrcROI.y), oDeviceSrc.pitch(), oROI, origin,
        oDeviceDst.data(oSrcROI.x, oSrcROI.y), oDeviceDst.pitch(), oROI,
        NPP_BORDER_REPLICATE));
}

void erodeROI(npp::ImageNPP_8u_C4 &oDeviceSrc, NppiRect oSrcROI, npp::ImageNPP_8u_C4 &oDeviceDst)
{
    NppiPoint origin = {0,0};
    NppiSize oROI = {oSrcROI.width, oSrcROI.height};

    NPP_CHECK_NPP(nppiErode3x3Border_8u_C4R(
        oDeviceSrc.data(oSrcROI.x, oSrcROI.y), oDeviceSrc.pitch(), oROI, origin,
        oDeviceDst.data(oSrcROI.x, oSrcROI.y), oDeviceDst.pitch(), oROI,
        NPP_BORDER_REPLICATE));
}

std::string mutateImage(npp::ImageNPP_8u_C4 &oDeviceSrc, npp::ImageNPP_8u_C4 &oDeviceDst)
{
    // Does 1 random mutation to `oDeviceSrc` and outputs to `oDeviceDst`.
    // Mutations include but are not limited to:
    // applying a texture, applying a filter, transforming the image geometry, or morpology.
    // The returned string describes the mutation done.

    NppiRect textureROI = {0, 0, 200, 100};
    NppiPoint textureStart = {100, 100};
    textureROI = moveROI(textureROI, textureStart);
    // addTextureROI(oDeviceSrc, textureROI, oRufflesTexture, oDeviceSrc);
    // addTextureROI(oDeviceSrc, textureROI, oRufflesTexture, oDeviceSrc);
    // addTextureROI(oDeviceSrc, textureROI, oRufflesTexture, oDeviceSrc);
    // addTextureROI(oDeviceSrc, textureROI, oRufflesTexture, oDeviceSrc);
    // addTextureROI(oDeviceSrc, textureROI, oRufflesTexture, oDeviceSrc);
    addTextureROI(oDeviceSrc, textureROI, oArgyleTexture, oDeviceSrc);

    // addTextureMTQ(oDeviceSrc, oArgyleTexture, oDeviceSrc);
    // addTextureTQ(oDeviceSrc, oRufflesTexture, oDeviceSrc);
    // addTextureTQ(oDeviceSrc, oRufflesTexture, oDeviceSrc);
    // addTextureTQ(oDeviceSrc, oRufflesTexture, oDeviceSrc);
    // addTextureTQ(oDeviceSrc, oRufflesTexture, oDeviceSrc);

    // addTextureBQ(oDeviceSrc, oRufflesTexture, oDeviceSrc);
    // addTextureBQ(oDeviceSrc, oRufflesTexture, oDeviceSrc);
    // addTextureBQ(oDeviceSrc, oRufflesTexture, oDeviceSrc);
    // addTextureBQ(oDeviceSrc, oRufflesTexture, oDeviceSrc);
    addTexture(oDeviceSrc, oRufflesTexture, oDeviceSrc);

    blurROI(oDeviceSrc, textureROI, oDeviceSrc);
    blurROI(oDeviceSrc, textureROI, oDeviceSrc);
    blurROI(oDeviceSrc, textureROI, oDeviceSrc);
    blurROI(oDeviceSrc, textureROI, oDeviceSrc);

    // approximate linear gradient between 0 and 255 split into 10 parts.
    // 0, 127, 255 excluded.
    Npp8u linear10[8] = {24, 51, 76, 102, 153, 179, 204, 230};

    // approximate linear gradient between 0 and 255 split into 5 parts.
    // 0, 127, 255 excluded.
    Npp8u linear5[4] = {24, 102, 153, 230};

    Npp8u constant[8] = {255, 255, 255, 255, 255, 255, 255, 255};
    Npp8u zeros[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    Npp8u halfs[8] = {127, 127, 127, 127, 127, 127, 127, 127};

    const Npp8u *pallet3[3] = {linear10, linear10, halfs};
    const Npp8u *pallet2[3] = {linear5, linear5, halfs};

    downSampleA3(oDeviceSrc, pallet3, oDeviceDst);
    // downSampleA2(oDeviceSrc, pallet2, oDeviceDst);

    // default gradient at 3-bit depth: 3000
    float cc = contourCount(oDeviceDst);
    std::cout << cc << std::endl;

    return "test";
}

void loadTextures()
{
    loadImage("data/textures/argyle.png", oArgyleTexture);
    loadImage("data/textures/crisp-paper-ruffles.png", oRufflesTexture);

    rotateTexture(oRufflesTexture, 45, oRufflesTexture);
}

int main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);
    loadTextures();

    // Base gradient all operations will be performed on.
    npp::ImageNPP_8u_C4 oDeviceGradient(500, 500);
    makeGradient(oDeviceGradient);
    npp::ImageNPP_8u_C4 oDeviceDst(oDeviceGradient.width(), oDeviceGradient.height());

    mutateImage(oDeviceGradient, oDeviceDst);

    std::string sResultFilename = "data/testG.png";
    npp::saveImage(sResultFilename, oDeviceDst);
    std::cout << "Saved image: " << sResultFilename << std::endl;

    nppiFree(oDeviceDst.data());
    nppiFree(oDeviceGradient.data());

    exit(EXIT_SUCCESS);
}
