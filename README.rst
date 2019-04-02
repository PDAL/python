================================================================================
PDAL
================================================================================

The PDAL Python extension allows you to process data with PDAL into `Numpy`_
arrays. Additionally, you can use it to fetch `schema`_ and `metadata`_ from
PDAL operations.

The repository for PDAL's Python extension is available at https://github.com/PDAL/python

It is released independently from PDAL itself as of PDAL 1.7.

Usage
--------------------------------------------------------------------------------

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
    pipeline.validate() # check if our JSON and options were good
    pipeline.loglevel = 8 #really noisy
    count = pipeline.execute()
    arrays = pipeline.arrays
    metadata = pipeline.metadata
    log = pipeline.log


.. _`Numpy`: http://www.numpy.org/
.. _`schema`: http://www.pdal.io/dimensions.html
.. _`metadata`: http://www.pdal.io/development/metadata.html


.. image:: https://travis-ci.org/PDAL/python.svg?branch=master
    :target: https://travis-ci.org/PDAL/python

.. image:: https://ci.appveyor.com/api/projects/status/of4kecyahpo8892d
   :target: https://ci.appveyor.com/project/hobu/python/

Requirements
================================================================================

* PDAL 1.7+
* Python >=2.7 (including Python 3.x)
* Cython (eg :code:`pip install cython`)
* Packaging (eg :code:`pip install packaging`)

