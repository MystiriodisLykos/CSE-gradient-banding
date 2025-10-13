
#ifndef NPP_IMAGE_TRANSLATIONS_H
#define NPP_IMAGE_TRANSLATIONS_H

#pragma once
// #include <npp.h>
#include <ImagesNPP.h>
// void rotateT(npp::ImageNPP_8u_C4 &oDeviceSrc, double angle, NppiPoint shift, npp::ImageNPP_8u_C4 &oDeviceDst);
void rotate(npp::ImageNPP_8u_C4 &oDeviceSrc, double angle, npp::ImageNPP_8u_C4 &oDeviceDst);
void crop(npp::ImageNPP_8u_C4 &oDeviceSrc, NppiPoint oCropPoint, npp::ImageNPP_8u_C4 &oDeviceDst);
void move(npp::ImageNPP_8u_C4 &oDeviceSrc, NppiPoint to, npp::ImageNPP_8u_C4 &oDeviceDst);

#endif // NPP_IMAGE_TRANSLATIONS_H