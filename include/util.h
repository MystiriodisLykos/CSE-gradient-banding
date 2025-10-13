
#ifndef NV_UTIL_NPP_IMAGE_IO_H
#define NV_UTIL_NPP_IMAGE_IO_H

#pragma once

#include "ImagesCPU.h"
#include "ImagesNPP.h"

#include "FreeImage.h"
#include "Exceptions.h"

#include <string>
#include "string.h"


// Error handler for FreeImage library.
//  In case this handler is invoked, it throws an NPP exception.
void FreeImageErrorHandler(FREE_IMAGE_FORMAT oFif, const char *zMessage)
{
    throw npp::Exception(zMessage);
}

// load and save image functions adapted from adapted from Common/ImageIO.h for pngs.

namespace npp
{
    // load a color image
    void
    loadImage(const std::string &rFileName, ImageCPU_8u_C4 &rImage)
    {
        // set your own FreeImage error handler
        FreeImage_SetOutputMessage(FreeImageErrorHandler);

        FREE_IMAGE_FORMAT eFormat = FreeImage_GetFileType(rFileName.c_str());

        // no signature? try to guess the file format from the file extension
        if (eFormat == FIF_UNKNOWN)
        {
            eFormat = FreeImage_GetFIFFromFilename(rFileName.c_str());
        }

        NPP_ASSERT(eFormat != FIF_UNKNOWN);
        // check that the plugin has reading capabilities ...
        FIBITMAP *pBitmap;

        // std::cerr << FreeImage_GetFormatFromFIF(eFormat) << std::endl;
        if (FreeImage_FIFSupportsReading(eFormat))
        {
            pBitmap = FreeImage_Load(eFormat, rFileName.c_str(), PNG_DEFAULT);
        }

        NPP_ASSERT(pBitmap != 0);

        // create an ImageCPU to receive the loaded image data
        npp::ImageCPU_8u_C4 oImage(FreeImage_GetWidth(pBitmap), FreeImage_GetHeight(pBitmap));

        // Copy the FreeImage data into the new ImageCPU
        unsigned int nSrcPitch = FreeImage_GetPitch(pBitmap);
        const Npp8u *pSrcLine = FreeImage_GetBits(pBitmap) + nSrcPitch * (FreeImage_GetHeight(pBitmap) - 1);
        Npp8u *pDstLine = oImage.data();
        unsigned int nDstPitch = oImage.pitch();

        for (size_t iLine = 0; iLine < oImage.height(); ++iLine)
        {
            memcpy(pDstLine, pSrcLine, oImage.width() * 4);
            pSrcLine -= nSrcPitch;
            pDstLine += nDstPitch;
        }

        // swap the user given image with our result image, effecively
        // moving our newly loaded image data into the user provided shell
        oImage.swap(rImage);
    }

    // Save a color image to disk.
    void
    saveImage(const std::string &rFileName, const ImageCPU_8u_C4 &rImage)
    {
        // create the result image storage using FreeImage so we can easily
        // save
        FIBITMAP *pResultBitmap = FreeImage_Allocate(rImage.width(), rImage.height(), 8 * 4 /* bits per pixel */);
        NPP_ASSERT_NOT_NULL(pResultBitmap);
        unsigned int nDstPitch = FreeImage_GetPitch(pResultBitmap);
        Npp8u *pDstLine = FreeImage_GetBits(pResultBitmap) + nDstPitch * (rImage.height() - 1);
        const Npp8u *pSrcLine = rImage.data();
        unsigned int nSrcPitch = rImage.pitch();

        for (size_t iLine = 0; iLine < rImage.height(); ++iLine)
        {
            memcpy(pDstLine, pSrcLine, rImage.width() * 4);
            pSrcLine += nSrcPitch;
            pDstLine -= nDstPitch;
        }

        // now save the result image
        bool bSuccess;
        bSuccess = FreeImage_Save(FIF_PNG, pResultBitmap, rFileName.c_str(), 0) == TRUE;
        NPP_ASSERT_MSG(bSuccess, "Failed to save result image.");
    }

    void
    saveImage(const std::string &rFileName, const ImageNPP_8u_C4 &rImage)
    {
        ImageCPU_8u_C4 oHostImage(rImage.size());
        // copy the device result data
        rImage.copyTo(oHostImage.data(), oHostImage.pitch());
        saveImage(rFileName, oHostImage);
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

} // namespace npp

#endif // NV_UTIL_NPP_IMAGE_IO_H
