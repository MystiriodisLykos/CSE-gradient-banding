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

void wrapTexture(npp::ImageNPP_8u_C4 &textureSrc, npp::ImageNPP_8u_C4 &oDeviceDSt) {
    NppiSize oTextureSizeROI = imageSizeROI(textureSrc);
    NppiSize oDeviceSizeROI = imageSizeROI(oDeviceDSt);

    nppiCopyWrapBorder_8u_C4R(
        textureSrc.data(), textureSrc.pitch(), oTextureSizeROI,
        oDeviceDSt.data(), oDeviceDSt.pitch(), oDeviceSizeROI,
        oDeviceSizeROI.height - oTextureSizeROI.height,
        oDeviceSizeROI.width - oTextureSizeROI.width);
}

void addTexture(npp::ImageNPP_8u_C4 &oDeviceSrc, npp::ImageNPP_8u_C4 &textureSrc, npp::ImageNPP_8u_C4 &oDeviceDSt)
{
    NppiSize oSrcSizeROI = imageSizeROI(oDeviceSrc);

    npp::ImageNPP_8u_C4 textureWrap(oDeviceDSt.width(), oDeviceDSt.height());
    wrapTexture(textureSrc, textureWrap);

    NPP_CHECK_NPP(nppiAdd_8u_C4RSfs(
        oDeviceSrc.data(), oDeviceSrc.pitch(),
        textureWrap.data(), textureWrap.pitch(),
        oDeviceDSt.data(), oDeviceDSt.pitch(),
        oSrcSizeROI, 0));
}

void makeGradient(npp::ImageNPP_8u_C4 &oDeviceGradient) {
    
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

    // Use resize linear interpolation to make a smooth gradient.
    NPP_CHECK_NPP(nppiResize_8u_C4R(
        oHostGradient.data(), oHostGradient.pitch(), oROI, oRectROI,
        oDeviceGradient.data(), oDeviceGradient.pitch(), oDeviceSizeROI, oDeviceRectROI,
        NPPI_INTER_LINEAR));
}

int main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);

    npp::ImageNPP_8u_C4 oDeviceGradient(1000,1000);
    makeGradient(oDeviceGradient);

    npp::ImageNPP_8u_C4 oDeviceTextureSrc;
    loadImage("data/textures/argyle.png", oDeviceTextureSrc);
    // loadImage("data/textures/crisp-paper-ruffles.png", oDeviceTextureSrc);

    npp::ImageNPP_8u_C4 oDeviceDst(oDeviceGradient.width(), oDeviceGradient.height());
    addTexture(oDeviceGradient, oDeviceTextureSrc, oDeviceDst);

    std::string sResultFilename = "data/testG.png";
    npp::saveImage(sResultFilename, oDeviceDst);
    std::cout << "Saved image: " << sResultFilename << std::endl;

    nppiFree(oDeviceDst.data());
    nppiFree(oDeviceGradient.data());

    exit(EXIT_SUCCESS);
}
