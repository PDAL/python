import unittest
import json
import pdal
import numpy
import os

DATADIRECTORY = "./test/data"


def a_filter(ins, outs):
    return True


class TestPythonInPython(unittest.TestCase):
    def test_pipeline_construction(self):

        pipeline = [
            os.path.join(DATADIRECTORY, "autzen-utm.las"),
            {
                "type": "filters.python",
                "script": __file__,
                "function": "a_filter",
                "module": "anything",
            },
        ]
        pdal.Pipeline(json.dumps(pipeline)).execute()
