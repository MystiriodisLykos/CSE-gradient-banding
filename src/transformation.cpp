// #include <npp.h>
// #include <ImagesNPP.h>
// #include <util.h>
#include <transformation.h>

// #include <helper_cuda.h>

// void rotateT(npp::ImageNPP_8u_C4 &oDeviceSrc, double angle, NppiPoint shift, npp::ImageNPP_8u_C4 &oDeviceDst)
// {
//     // rotates oDeviceSrc by `angle` and translated by `shift`, reults in `oDeviceDSt`.

//     // NPP_CHECK_NPP(nppiRotate_8u_C4R(
//     //     oDeviceSrc.data(), npp::imageSizeROI(oDeviceSrc), oDeviceSrc.pitch(), npp::imageROI(oDeviceSrc),
//     //     oDeviceDst.data(), oDeviceDst.pitch(), npp::imageROI(oDeviceDst),
//     //     angle, -shift.x, -shift.y,
//     //     // angle, 0, 0,
//     //     NPPI_INTER_LINEAR));
// }

void rotate(npp::ImageNPP_8u_C4 &oDeviceSrc, double angle, npp::ImageNPP_8u_C4 &oDeviceDst)
{
    // // Calculate rotated bounding box.
    // double aBoundingBox[2][2] = {0};
    // NPP_CHECK_NPP(nppiGetRotateBound(npp::imageROI(oDeviceSrc), aBoundingBox, angle, 0, 0));
    // NppiRect oRotatedROI = {0,
    //                         0,
    //                         (int)(aBoundingBox[1][0] - aBoundingBox[0][0]),
    //                         (int)(aBoundingBox[1][1] - aBoundingBox[0][1])};

    // npp::ImageNPP_8u_C4 oRotated(oRotatedROI.width, oRotatedROI.height);

    // // shift to center rotated image.
    // NppiPoint shift = {(int)aBoundingBox[0][0], (int)aBoundingBox[0][1]};
    // rotateT(oDeviceSrc, angle, shift, oRotated);

    // oDeviceDst.swap(oRotated);
}

void crop(npp::ImageNPP_8u_C4 &oDeviceSrc, NppiPoint oCropPoint, npp::ImageNPP_8u_C4 &oDeviceDst)
{
    // crops `oDeviceSrc` from `oCropPoint` to size of `oDeviceDst`, result in `oDeviceDst`.

    // There doesn't seem to be an easy crop function, so a 0 degree rotation is used instead.
    // rotateT(oDeviceSrc, 0, oCropPoint, oDeviceDst);
}

void move(npp::ImageNPP_8u_C4 &oDeviceSrc, NppiPoint to, npp::ImageNPP_8u_C4 &oDeviceDst)
{
    // Moves `oDeviceSrc` to the point `to` in `oDeviceDst`.

    // A 0 degree rotation feels like the easiest way to do this.
    // NPP_CHECK_NPP(nppiRotate_8u_C4R(
    //     oDeviceSrc.data(), npp::imageSizeROI(oDeviceSrc), oDeviceSrc.pitch(), npp::imageROI(oDeviceSrc),
    //     oDeviceDst.data(), oDeviceDst.pitch(), npp::imageROI(oDeviceDst),
    //     0, to.x, to.y,
    //     NPPI_INTER_LINEAR));
}
// int main(int argc, char *argv[]) { return 0; }