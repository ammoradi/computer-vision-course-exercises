Document Reader
---------------
this program takes an image containing a document(paper)
as input and returns cropped and perspective transformed
document(paper) as output by following steps:

* read input image
* detect edges using `Canny Edge Detector`
* find maximum value **Contours** to finding border of document.
* crop the document by finding four corners of document polygon.
* map the four corners by `Perspective Transform` to create document's rectangle.
* show and save the rectangle result

NOTE: _uncomment lines 109 and 110 of `main.py` to have scanned like document_
