Image Stitching
---------------
this program gets input image's name and count to read all particle images.
for example:

```
img_name = "yard"
img_count = 9
```
will read `./Images/yard-00.png` to `./Images/yard-09.png` images.

then using opencv's built in `Stitching` method will generate **mosaic** image.

Reference
---------
* [https://docs.opencv.org/4.0.0-rc/d5/d48/samples_2python_2stitching_8py-example.html](https://docs.opencv.org/4.0.0-rc/d5/d48/samples_2python_2stitching_8py-example.html)