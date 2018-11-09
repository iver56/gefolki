GeFolki is a coregistration software that has been developped in the framework of the MEDUSA project, first for SAR/SAR co-registration, then for other cases of remote sensing image coregistration, including hetergeneous image co-registration (ex LIDAR/SAR, optics/SAR, hyperspectral/optics and so on)

We hope this code will be helpful and we remain to your disposal and we are interested in getting back your remarks.

## Citations for {ge ,e, _}Folki algorithms :

If you are using the result of GeFolki in your project, we kindly ask you to cite :

- for SAR/SAR coregistration (interferometry, change detection, and so on)

Aurélien Plyer, Elise Colin-Koeniguer, Flora Weissgerber, "A New Coregistration Algorithm for Recent Applications on Urban SAR Images", Geoscience and Remote Sensing Letters, IEEE , vol.12, no.11, pp. 2198 – 2202, nov 2015

- for other geosensing cases (optics/SAR, optics/hyperspectral, LIDAR/SAR, etc.)

Guillaume Brigot, Elise Colin-Koeniguer, Aurélien Plyer, Fabrice Janez, "Adaptation and Evaluation of an Optical Flow Method Applied to Coregistration of Forest Remote Sensing Images", IEEE Journal of Selected Topics in Applied Earth Observations ans Remote Sensing, Volume 9, Issue 7, July 2016

- for measurement applications (ex : PIV, material deformation, ...) : 

Champagnat, F., Plyer, A., Le Besnerais, G., Leclaire, B., & Le Sant, Y. (2009, August). How to calculate dense piv vector fields at video rate. In Proceedings of 8th International Symposium on Particle Image Velocimetry-PIV09 (Vol. 11, pp. 15-20).

- for computer vision (ex : robotics) :

Plyer, A., Le Besnerais, G., & Champagnat, F. (2016). Massively parallel Lucas Kanade optical flow for real-time video processing applications. Journal of Real-Time Image Processing, 11(4), 713-730.


## help

The source code is provided here in python under a GPL license.

After decompression, this directory contains:

- a directory "datasets" that are used in the demonstration file "demo.py". Origins of the different datasets are described in the readme.txt
- the different python source codes. The demonstration are proposed in the demo.py file.

In python you find a part of folki optical flow familly (folki, efolki, gefolki).
