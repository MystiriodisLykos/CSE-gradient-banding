#include <Exceptions.h>
#include <util.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>

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
        (int)oDeviceSrc.height()
        };
    return oROI;
}

NppiRect imageROI(npp::ImageNPP_8u_C4 &oDeviceSrc) {
    NppiSize oSizeROI = imageSizeROI(oDeviceSrc);
    NppiRect oROI = {
        0,
        0,
        oSizeROI.width,
        oSizeROI.height};
    return oROI;
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
        NPPI_INTER_NN));
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

void addTextureROI(npp::ImageNPP_8u_C4 &oDeviceSrc, NppiRect oSrcROI, npp::ImageNPP_8u_C4 &textureSrc, npp::ImageNPP_8u_C4 &oDeviceDSt)
{
    // Adds the textureSrc to the oDeviceSrc only in the oSrcROI, outputs to the oDeivceDst.
    // textureSrc is repeated vertically and horizontally as needed to cover the oSrcROI.

    // Crop oDeviceSrc to the oSrcROI
    npp::ImageNPP_8u_C4 oDeviceRegion(oSrcROI.width, oSrcROI.height);
    NppiPoint start = {oSrcROI.x, oSrcROI.y};
    crop(oDeviceSrc, start, oDeviceRegion);

    addTexture(oDeviceRegion, textureSrc, oDeviceRegion);

    // Make a copy of oDeviceSrc to coped the textured ROI into.
    npp::ImageNPP_8u_C4 oResImage(oDeviceSrc.width(), oDeviceSrc.height());
    nppiCopy_8u_C4R(
        oDeviceSrc.data(), oDeviceSrc.pitch(),
        oResImage.data(), oResImage.pitch(),
        imageSizeROI(oDeviceSrc));

    // Move the textured region on the copy of the src iamge.
    move(oDeviceRegion, start, oResImage);

    oDeviceDSt.swap(oResImage);
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
    npp::ImageNPP_8u_C4 oHostGradient(2,4);

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

void downSampleA3(npp::ImageNPP_8u_C4 &oDeviceSrc, const Npp8u *pTables[3], npp::ImageNPP_8u_C4 &oDeviceDst)

{
    // Downsample `oDeviceSrc` using `pTables` to 8 values in rgb channels, alpha is not downsampled.

    Npp8u whites[8] = {255, 255, 255, 255, 255, 255, 255, 255};

    Npp8u *d_whites = NULL;
    NPP_CHECK_CUDA(cudaMalloc(&d_whites, 8 * sizeof(Npp8u)));
    NPP_CHECK_CUDA(cudaMemcpy(d_whites, whites, 8 * sizeof(Npp8u), cudaMemcpyHostToDevice));

    // npp::ImageNPP_8u_C4 oDeviceInv(oDeviceSrc.width(), oDeviceSrc.height());

    // NPP_CHECK_NPP(nppiNot_8u_AC4R(
    //     oDeviceSrc.data(), oDeviceSrc.pitch(),
    //     oDeviceInv.data(), oDeviceInv.pitch(),
    //     imageSizeROI(oDeviceInv)));

    // NPP_CHECK_NPP(nppiLUTPalette_8u_AC4R(
    //     oDeviceInv.data(), oDeviceInv.pitch(),
    //     oDeviceInv.data(), oDeviceInv.pitch(),
    //     imageSizeROI(oDeviceSrc),
    //     pTables, 3));

    // NPP_CHECK_NPP(nppiSet_8u_C4CR(
    //     255, oDeviceInv.data() + 3, oDeviceInv.pitch(), imageSizeROI(oDeviceInv)));

    // oDeviceDst.swap(oDeviceInv);
    const Npp32u aConstants[4] = {5,5,0,0};
    NPP_CHECK_NPP(nppiRShiftC_8u_C4R(
        oDeviceSrc.data(), oDeviceSrc.pitch(), aConstants,
        oDeviceDst.data(), oDeviceDst.pitch(), imageSizeROI(oDeviceSrc)));

    NPP_CHECK_NPP(nppiLUTPalette_8u_AC4R(
        oDeviceDst.data(), oDeviceDst.pitch(),
        oDeviceDst.data(), oDeviceDst.pitch(),
        imageSizeROI(oDeviceSrc),
        pTables, 3));

    NPP_CHECK_NPP(nppiLShiftC_8u_C4R(
        oDeviceDst.data(), oDeviceDst.pitch(), aConstants,
        oDeviceDst.data(), oDeviceDst.pitch(), imageSizeROI(oDeviceSrc)));

    // const Npp8u *pallet[4] = {pTables[0], pTables[1], pTables[2], d_whites};
    // // const Npp8u *pallet[4] = {d_whites, d_whites, d_whites, d_whites};
    // npp::ImageNPP_8u_C4 oDeviceHSV(oDeviceSrc.width(), oDeviceSrc.height());
    // npp::ImageNPP_8u_C4 oDeviceHSVR(oDeviceSrc.width(), oDeviceSrc.height());
    // npp::ImageNPP_8u_C4 oDeviceRGB(oDeviceSrc.width(), oDeviceSrc.height());

    // NPP_CHECK_NPP(nppiRGBToHSV_8u_AC4R(
    //     oDeviceSrc.data(), oDeviceSrc.pitch(),
    //     oDeviceHSV.data(), oDeviceHSV.pitch(),
    //     imageSizeROI(oDeviceSrc)));

    // NPP_CHECK_NPP(nppiLUTPalette_8u_AC4R(
    //     oDeviceHSV.data(), oDeviceHSV.pitch(),
    //     oDeviceHSVR.data(), oDeviceHSVR.pitch(),
    //     imageSizeROI(oDeviceSrc),
    //     pTables, 3));

    // NPP_CHECK_NPP(nppiHSVToRGB_8u_AC4R(
    //     oDeviceHSVR.data(), oDeviceHSVR.pitch(),
    //     oDeviceRGB.data(), oDeviceRGB.pitch(),
    //     imageSizeROI(oDeviceSrc)));

    // NPP_CHECK_NPP(nppiSet_8u_C4CR(
    //     255, oDeviceRGB.data() + 3, oDeviceRGB.pitch(), imageSizeROI(oDeviceSrc)));

    // oDeviceDst.swap(oDeviceRGB);
    // std::cout << "here" << std::endl;
}

int
main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);

    npp::ImageNPP_8u_C4 oDeviceLena;
    loadImage("data/Lena.png", oDeviceLena);

    npp::ImageNPP_8u_C4 oDeviceGradient(2000,2000);
    makeGradient(oDeviceGradient);

    npp::ImageNPP_8u_C4 oDeviceTextureSrc;
    // loadImage("data/textures/argyle.png", oDeviceTextureSrc);
    loadImage("data/textures/crisp-paper-ruffles.png", oDeviceTextureSrc);

    // npp::ImageNPP_8u_C4 oDeviceDst(oDeviceGradient.width(), oDeviceGradient.height());
    // addTexture(oDeviceGradient, oDeviceTextureSrc, oDeviceDst);

    rotateTexture(oDeviceTextureSrc, 40, oDeviceTextureSrc);

    // npp::ImageNPP_8u_C4 oDeviceDst(oDeviceLena.width(), oDeviceLena.height());
    npp::ImageNPP_8u_C4 oDeviceDst(oDeviceGradient.width(), oDeviceGradient.height());
    NppiPoint shift = {100, 500};
    // rotateT(oDeviceLena, 0, shift, oDeviceDst);

    // crop(oDeviceLena, shift, oDeviceDst);

    // rotateTexture(oDeviceLena, 45.0, oDeviceDst);

    NppiRect textureROI = {100, 500, 500, 500};
    // addTextureROI(oDeviceGradient, textureROI, oDeviceTextureSrc, oDeviceDst);
    // addTextureROI(oDeviceDst, textureROI, oDeviceTextureSrc, oDeviceDst);
    // addTextureROI(oDeviceDst, textureROI, oDeviceTextureSrc, oDeviceDst);
    // addTextureROI(oDeviceDst, textureROI, oDeviceTextureSrc, oDeviceDst);
    // addTextureROI(oDeviceDst, textureROI, oDeviceTextureSrc, oDeviceDst);
    // addTextureROI(oDeviceDst, textureROI, oDeviceTextureSrc, oDeviceDst);

    // addTexture(oDeviceGradient, oDeviceTextureSrc, oDeviceGradient);
    // addTexture(oDeviceGradient, oDeviceTextureSrc, oDeviceGradient);
    // addTexture(oDeviceGradient, oDeviceTextureSrc, oDeviceGradient);
    // addTexture(oDeviceGradient, oDeviceTextureSrc, oDeviceGradient);
    // addTexture(oDeviceGradient, oDeviceTextureSrc, oDeviceGradient);
    // addTexture(oDeviceGradient, oDeviceTextureSrc, oDeviceGradient);

    // 10011000
    // 10110011

    // approximate linear gradient between 0 and 255 split into 10 parts.
    // 0, 127, 255 excluded.
    // TODO: needs to be a device pointer
    // Npp8u linear10[8] = {24, 51, 76, 102, 153, 179, 204, 230};
    // TODO: this needs to be in binary
    Npp8u linear10[8] = {255, 0, 51, 179, 76, 204, 102, 0};
    // Npp8u linear10[8] = {230, 204, 179, 153, 102, 76, 51, 24};
    Npp8u constant[8] = {255,255,255,255,255,255,255,255};
    Npp8u zeros[8] = {0,0,0,0,0,0,0,0};

    size_t p10_size = 8 * sizeof(Npp8u);
    Npp8u *d_linear10 = NULL;
    NPP_CHECK_CUDA(cudaMalloc(&d_linear10, p10_size));
    NPP_CHECK_CUDA(cudaMemcpy(d_linear10, linear10, p10_size, cudaMemcpyHostToDevice));

    Npp8u *d_constant = NULL;
    NPP_CHECK_CUDA(cudaMalloc(&d_constant, p10_size));
    NPP_CHECK_CUDA(cudaMemcpy(d_constant, constant, p10_size, cudaMemcpyHostToDevice));

    Npp8u *d_zeros = NULL;
    NPP_CHECK_CUDA(cudaMalloc(&d_zeros, p10_size));
    NPP_CHECK_CUDA(cudaMemcpy(d_zeros, zeros, p10_size, cudaMemcpyHostToDevice));

    const Npp8u *pallet[3] = {d_linear10, d_linear10, d_constant}; // blue, green, red

    downSampleA3(oDeviceGradient, pallet, oDeviceDst);

    std::string sResultFilename = "data/testG.png";
    npp::saveImage(sResultFilename, oDeviceDst);
    std::cout << "Saved image: " << sResultFilename << std::endl;

    nppiFree(oDeviceDst.data());
    nppiFree(oDeviceGradient.data());

    exit(EXIT_SUCCESS);
}
