# shape_icc
***
Python implementation of Smith and Smith's Shape ICC method for similiarity measurement between segmentations.

Paper is here: https://doi.org/10.1371/journal.pone.0202087
 
***
Bibtex citation is available here
```
@article{10.1371/journal.pone.0202087,
    author = {Smith, Travis B. AND Smith, Ning},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {Agreement and reliability statistics for shapes},
    year = {2018},
    month = {08},
    volume = {13},
    url = {https://doi.org/10.1371/journal.pone.0202087},
    pages = {1-11},
    abstract = {We describe a methodology for assessing agreement and reliability among a set of shapes. Motivated by recent studies of the reliability of manually segmented medical images, we focus on shapes composed of rasterized, binary-valued data representing closed geometric regions of interest. The methodology naturally generalizes to N dimensions and other data types, though. We formulate the shape variance, shape correlation and shape intraclass correlation coefficient (ICC) in terms of a simple distance metric, the Manhattan norm, which quantifies the absolute difference between any two shapes. We demonstrate applications of this methodology by working through example shape variance calculations in 1-D, for the analysis of overlapping line segments, and 2-D, for the analysis of overlapping regions. We also report the results of a simulated reliability analysis of manually delineated shape boundaries, and we compare the shape ICC with the more conventional and commonly used area ICC. The proposed shape-sensitive methodology captures all of the variation in the shape measurements, and it provides a more accurate estimate of the measurement reliability than an analysis of only the measured areas.},
    number = {8},
    doi = {10.1371/journal.pone.0202087}
}
```

