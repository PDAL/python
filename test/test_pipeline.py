import json
import os
import sys

import numpy as np
import pytest

import pdal

DATADIRECTORY = os.path.join(os.path.dirname(__file__), "data")


def get_pipeline(filename, factory=pdal.Pipeline):
    with open(os.path.join(DATADIRECTORY, filename), "r") as f:
        pipeline = factory(f.read())
    assert pipeline.validate()
    return pipeline


class TestPipeline:
    def test_construction(self):
        """Can we construct a PDAL pipeline"""
        assert isinstance(get_pipeline("sort.json"), pdal.Pipeline)

    def test_execution(self):
        """Can we execute a PDAL pipeline"""
        r = get_pipeline("sort.json")
        r.execute()
        assert len(r.pipeline) > 200

    def test_validate(self):
        """Do we complain with bad pipelines"""
        bad_json = """
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
        r = pdal.Pipeline(bad_json)
        with pytest.raises(RuntimeError):
            r.validate()

    def test_array(self):
        """Can we fetch PDAL data as a numpy array"""
        r = get_pipeline("sort.json")
        r.execute()
        arrays = r.arrays
        assert len(arrays) == 1

        a = arrays[0]
        assert a[0][0] == 635619.85
        assert a[1064][2] == 456.92

    def test_metadata(self):
        """Can we fetch PDAL metadata"""
        r = get_pipeline("sort.json")
        with pytest.raises(RuntimeError):
            r.metadata
        r.execute()
        j = json.loads(r.metadata)
        assert j["metadata"]["readers.las"][0]["count"] == 1065

    def test_schema(self):
        """Fetching a schema works"""
        r = get_pipeline("sort.json")
        with pytest.raises(RuntimeError):
            r.schema
        r.execute()
        assert r.schema["schema"]["dimensions"][0]["name"] == "X"

    def test_no_execute(self):
        """Does fetching arrays without executing throw an exception"""
        r = get_pipeline("sort.json")
        with pytest.raises(RuntimeError):
            r.arrays

    def test_merged_arrays(self):
        """Can we fetch multiple point views from merged PDAL data"""
        r = get_pipeline("chip.json")
        r.execute()
        arrays = r.arrays
        assert len(arrays) == 43

    # def test_logging(self):
    #    """Can we fetch log output"""
    #    r = get_pipeline('reproject.json')
    #    count = r.execute()
    #    assert count == 789
    #    assert r.log.split()[0] == '(pypipeline')


class TestArrayLoad:
    def test_merged_arrays(self):
        """Can we load data from a list of arrays to PDAL"""
        data = np.load(os.path.join(DATADIRECTORY, "test3d.npy"))
        arrays = [data, data, data]
        filter_intensity = """{
          "pipeline":[
            {
              "type":"filters.range",
              "limits":"Intensity[100:300)"
            }
          ]
        }"""
        p = pdal.Pipeline(filter_intensity, arrays)
        p.execute()
        arrays = p.arrays
        assert len(arrays) == 3

        for data in arrays:
            assert len(data) == 12
            assert data["Intensity"].sum() == 1926

    def test_read_arrays(self):
        """Can we read and filter data from a list of arrays to PDAL"""
        # just some dummy data
        x_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_vals = [6.0, 7.0, 8.0, 9.0, 10.0]
        z_vals = [1.5, 3.5, 5.5, 7.5, 9.5]
        test_data = np.array(
            [(x, y, z) for x, y, z in zip(x_vals, y_vals, z_vals)],
            dtype=[("X", np.float), ("Y", np.float), ("Z", np.float)],
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
        p = pdal.Pipeline(pipeline, arrays=[test_data])
        count = p.execute()
        arrays = p.arrays
        assert count == 2
        assert len(arrays) == 1

    def test_reference_counting(self):
        """Can we read and filter data from a list of arrays to PDAL"""
        # just some dummy data
        x_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_vals = [6.0, 7.0, 8.0, 9.0, 10.0]
        z_vals = [1.5, 3.5, 5.5, 7.5, 9.5]
        test_data = np.array(
            [(x, y, z) for x, y, z in zip(x_vals, y_vals, z_vals)],
            dtype=[("X", np.float), ("Y", np.float), ("Z", np.float)],
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
        p = pdal.Pipeline(pipeline, arrays=[test_data])
        count = p.execute()
        assert count == 2
        refcount = sys.getrefcount(p.arrays[0])
        assert refcount == 1


class TestDimensions:
    def test_fetch_dimensions(self):
        """Ask PDAL for its valid dimensions list"""
        dims = pdal.dimensions
        assert 71 < len(dims) < 120


class TestMesh:
    def test_no_execute(self):
        """Does fetching meshes without executing throw an exception"""
        r = get_pipeline("sort.json")
        with pytest.raises(RuntimeError):
            r.meshes

    def test_mesh(self):
        """Can we fetch PDAL face data as a numpy array"""
        r = get_pipeline("mesh.json")
        points = r.execute()
        assert points == 1065
        meshes = r.meshes
        assert len(meshes) == 24

        m = meshes[0]
        assert str(m.dtype) == "[('A', '<u4'), ('B', '<u4'), ('C', '<u4')]"
        assert len(m) == 134
        assert m[0][0] == 29
