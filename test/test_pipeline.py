import unittest
import pdal
import os
import numpy as np
from packaging.version import Version

DATADIRECTORY = "./test/data"

bad_json = u"""
{
  "pipeline": [
    "nofile.las",
    {
        "type": "filters.sort",
        "dimension": "X"
    }
  ]
}
"""



class PDALTest(unittest.TestCase):

    def fetch_json(self, filename):
        import os
        fn = DATADIRECTORY + os.path.sep +  filename
        output = ''
        with open(fn, 'rb') as f:
            output = f.read().decode('UTF-8')
        return output

class TestPipeline(PDALTest):

    @unittest.skipUnless(os.path.exists(os.path.join(DATADIRECTORY, 'sort.json')),
                         "missing test data")
    def test_construction(self):
        """Can we construct a PDAL pipeline"""
        json = self.fetch_json('sort.json')
        r = pdal.Pipeline(json)

    @unittest.skipUnless(os.path.exists(os.path.join(DATADIRECTORY, 'sort.json')),
                         "missing test data")
    def test_execution(self):
        """Can we execute a PDAL pipeline"""
        x = self.fetch_json('sort.json')
        r = pdal.Pipeline(x)
        r.validate()
        r.execute()
        self.assertGreater(len(r.pipeline), 200)

    def test_validate(self):
        """Do we complain with bad pipelines"""
        r = pdal.Pipeline(bad_json)
        with self.assertRaises(RuntimeError):
            r.validate()

    @unittest.skipUnless(os.path.exists(os.path.join(DATADIRECTORY, 'sort.json')),
                         "missing test data")
    def test_array(self):
        """Can we fetch PDAL data as a numpy array"""
        json = self.fetch_json('sort.json')
        r = pdal.Pipeline(json)
        r.validate()
        r.execute()
        arrays = r.arrays
        self.assertEqual(len(arrays), 1)

        a = arrays[0]
        self.assertAlmostEqual(a[0][0], 635619.85, 7)
        self.assertAlmostEqual(a[1064][2], 456.92, 7)

    @unittest.skipUnless(os.path.exists(os.path.join(DATADIRECTORY, 'sort.json')),
                         "missing test data")
    def test_metadata(self):
        """Can we fetch PDAL metadata"""
        json = self.fetch_json('sort.json')
        r = pdal.Pipeline(json)
        r.validate()
        r.execute()
        metadata = r.metadata
        import json
        j = json.loads(metadata)
        self.assertEqual(j["metadata"]["readers.las"][0]["count"], 1065)


    @unittest.skipUnless(os.path.exists(os.path.join(DATADIRECTORY, 'sort.json')),
                         "missing test data")
    def test_no_execute(self):
        """Does fetching arrays without executing throw an exception"""
        json = self.fetch_json('sort.json')
        r = pdal.Pipeline(json)
        with self.assertRaises(RuntimeError):
            r.arrays
#
#    @unittest.skipUnless(os.path.exists(os.path.join(DATADIRECTORY, 'reproject.json')),
#                         "missing test data")
#    def test_logging(self):
#        """Can we fetch log output"""
#        json = self.fetch_json('reproject.json')
#        r = pdal.Pipeline(json)
#        r.loglevel = 8
#        r.validate()
#        count = r.execute()
#        self.assertEqual(count, 789)
#        self.assertEqual(r.log.split()[0], '(pypipeline')
#
    @unittest.skipUnless(os.path.exists(os.path.join(DATADIRECTORY, 'sort.json')),
                         "missing test data")
    def test_schema(self):
        """Fetching a schema works"""
        json = self.fetch_json('sort.json')
        r = pdal.Pipeline(json)
        r.validate()
        r.execute()
        self.assertEqual(r.schema['schema']['dimensions'][0]['name'], 'X')

    @unittest.skipUnless(os.path.exists(os.path.join(DATADIRECTORY, 'chip.json')),
                         "missing test data")
    def test_merged_arrays(self):
        """Can we fetch multiple point views from merged PDAL data """
        json = self.fetch_json('chip.json')
        r = pdal.Pipeline(json)
        r.validate()
        r.execute()
        arrays = r.arrays
        self.assertEqual(len(arrays), 43)


class TestArrayLoad(PDALTest):

    @unittest.skipUnless(os.path.exists(os.path.join(DATADIRECTORY, 'perlin.npy')),
            "missing test data")
    def test_merged_arrays(self):
        """Can we load data from a list of arrays to PDAL"""
        if Version(pdal.info.version) < Version('1.8'):
            return True
        data = np.load(os.path.join(DATADIRECTORY, 'test3d.npy'))

        arrays = [data, data, data]

        json = self.fetch_json('chip.json')
        chip =u"""{
  "pipeline":[
    {
      "type":"filters.range",
      "limits":"Intensity[100:300)"
    }
  ]
}"""

        p = pdal.Pipeline(chip, arrays)
        p.loglevel = 8
        count = p.execute()
        arrays = p.arrays
        self.assertEqual(len(arrays), 3)

        for data in arrays:
            self.assertEqual(len(data), 12)
            self.assertEqual(data['Intensity'].sum(), 1926)

    def test_read_arrays(self):
        """Can we read and filter data from a list of arrays to PDAL"""
        if Version(pdal.info.version) < Version('1.8'):
            return True

        # just some dummy data
        x_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_vals = [6.0, 7.0, 8.0, 9.0, 10.0]
        z_vals = [1.5, 3.5, 5.5, 7.5, 9.5]
        test_data = np.array(
            [(x, y, z) for x, y, z in zip(x_vals, y_vals, z_vals)],
            dtype=[('X', np.float), ('Y', np.float), ('Z', np.float)]
        )

        pipeline = """
        {
            "pipeline": [
                {
                    "type":"filters.range",
                    "limits":"X[2.5:4.5]"
                }
            ]
        }
        """

        p = pdal.Pipeline(pipeline, arrays=[test_data,])
        p.loglevel = 8
        count = p.execute()
        arrays = p.arrays
        self.assertEqual(count, 2)
        self.assertEqual(len(arrays), 1)


class TestDimensions(PDALTest):
    def test_fetch_dimensions(self):
        """Ask PDAL for its valid dimensions list"""
        dims = pdal.dimensions
        if Version(pdal.info.version) < Version('1.8'):
            self.assertEqual(len(dims), 71)
        else:
            self.assertEqual(len(dims), 72)

def test_suite():
    return unittest.TestSuite(
        [TestPipeline])

if __name__ == '__main__':
    unittest.main()
