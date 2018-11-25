What the program does?:
=============
* read static image from directory
* uses self-defined **otsu** Function two make image binary
* show binary image

How to run?:
=============
* run app.py file by following command: `$ python3.6 app.py`

Approaches:
=============
1. `Sw` = (`W1`*`S1`^`2`) + (`W2`*`S2`^`2`) 
2. `Sw` = `W1` * `W2` * (`S2` - `S1`)^`2`
we used second approach for better result.

Refrences:
=============
* [Otsu Thresholding](http://www.labbookpages.co.uk/software/imgProc/otsuThreshold.html)

Todo:
=============
its better to first denoise the image and then apply otsu algorithm.