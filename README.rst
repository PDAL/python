================================================================================
PDAL
================================================================================

PDAL Python support allows you to process data with PDAL into `Numpy`_ arrays.
It supports embedding Python in PDAL pipelines with the `readers.numpy
<https://pdal.io/stages/readers.numpy.html>`__ and `filters.python
<https://pdal.io/stages/filters.python.html>`__ stages, and it provides a PDAL
extension module to control Python interaction with PDAL.

Additionally, you can use it to fetch `schema`_ and `metadata`_ from
PDAL operations.

Installation
--------------------------------------------------------------------------------

PyPI
................................................................................

PDAL Python support is installable via PyPI:

.. code-block::

    pip install PDAL

GitHub
................................................................................

The repository for PDAL's Python extension is available at https://github.com/PDAL/python

Python support released independently from PDAL itself as of PDAL 1.7.

Usage
--------------------------------------------------------------------------------

Simple
................................................................................

Given the following pipeline, which simply reads an `ASPRS LAS`_ file and
sorts it by the ``X`` dimension:

.. _`ASPRS LAS`: https://www.asprs.org/committee-general/laser-las-file-format-exchange-activities.html

.. code-block:: python


    json = """
    {
      "pipeline": [
        "1.2-with-color.las",
        {
            "type": "filters.sort",
            "dimension": "X"
        }
      ]
    }"""

    import pdal
    pipeline = pdal.Pipeline(json)
    count = pipeline.execute()
    arrays = pipeline.arrays
    metadata = pipeline.metadata
    log = pipeline.log

Reading  using Numpy Arrays
................................................................................

The following more complex scenario demonstrates the full cycling between
PDAL and Python:

* Read a small testfile from GitHub into a Numpy array
* Filters those arrays with Numpy for Intensity
* Pass the filtered array to PDAL to be filtered again
* Write the filtered array to an LAS file.

.. code-block:: python

    data = "https://github.com/PDAL/PDAL/blob/master/test/data/las/1.2-with-color.las?raw=true"


    json = """
        {
          "pipeline": [
            {
                "type": "readers.las",
                "filename": "%s"
            }
          ]
        }"""

    import pdal
    import numpy as np
    pipeline = pdal.Pipeline(json % data)
    count = pipeline.execute()

    # get the data from the first array
    # [array([(637012.24, 849028.31, 431.66, 143, 1,
    # 1, 1, 0, 1,  -9., 132, 7326, 245380.78254963,  68,  77,  88),
    # dtype=[('X', '<f8'), ('Y', '<f8'), ('Z', '<f8'), ('Intensity', '<u2'),
    # ('ReturnNumber', 'u1'), ('NumberOfReturns', 'u1'), ('ScanDirectionFlag', 'u1'),
    # ('EdgeOfFlightLine', 'u1'), ('Classification', 'u1'), ('ScanAngleRank', '<f4'),
    # ('UserData', 'u1'), ('PointSourceId', '<u2'),
    # ('GpsTime', '<f8'), ('Red', '<u2'), ('Green', '<u2'), ('Blue', '<u2')])

    arr = pipeline.arrays[0]
    print (len(arr)) # 1065 points


    # Filter out entries that have intensity < 50
    intensity = arr[arr['Intensity'] > 30]
    print (len(intensity)) # 704 points


    # Now use pdal to clamp points that have intensity
    # 100 <= v < 300, and there are 387
    clamp =u"""{
      "pipeline":[
        {
          "type":"filters.range",
          "limits":"Intensity[100:300)"
        }
      ]
    }"""

    p = pdal.Pipeline(clamp, [intensity])
    count = p.execute()
    clamped = p.arrays[0]
    print (count)

    # Write our intensity data to an LAS file
    output =u"""{
      "pipeline":[
        {
          "type":"writers.las",
          "filename":"clamped.las",
          "offset_x":"auto",
          "offset_y":"auto",
          "offset_z":"auto",
          "scale_x":0.01,
          "scale_y":0.01,
          "scale_z":0.01
        }
      ]
    }"""

    p = pdal.Pipeline(output, [clamped])
    count = p.execute()
    print (count)




.. _`Numpy`: http://www.numpy.org/
.. _`schema`: http://www.pdal.io/dimensions.html
.. _`metadata`: http://www.pdal.io/development/metadata.html


.. image:: https://ci.appveyor.com/api/projects/status/of4kecyahpo8892d
   :target: https://ci.appveyor.com/project/hobu/python/

Requirements
================================================================================

* PDAL 2.1+
* Python >=3.6
* Cython (eg :code:`pip install cython`)
* Packaging (eg :code:`pip install packaging`)

