import json
import os
import sys

import numpy as np
import pytest

import pdal

DATADIRECTORY = os.path.join(os.path.dirname(__file__), "data")


def get_pipeline(filename, validate=True):
    with open(os.path.join(DATADIRECTORY, filename), "r") as f:
        if filename.endswith(".json"):
            pipeline = pdal.Pipeline(f.read())
        elif filename.endswith(".py"):
            pipeline = eval(f.read(), vars(pdal))
    if validate:
        assert pipeline.validate()
    return pipeline


class TestPipeline:
    @pytest.mark.parametrize("filename", ["sort.json", "sort.py"])
    def test_construction(self, filename):
        """Can we construct a PDAL pipeline"""
        assert isinstance(get_pipeline(filename), pdal.Pipeline)

    @pytest.mark.parametrize("filename", ["sort.json", "sort.py"])
    def test_execution(self, filename):
        """Can we execute a PDAL pipeline"""
        r = get_pipeline(filename)
        r.execute()
        assert len(r.pipeline) > 200

    @pytest.mark.parametrize("filename", ["bad.json", "bad.py"])
    def test_validate(self, filename):
        """Do we complain with bad pipelines"""
        r = get_pipeline(filename, validate=False)
        with pytest.raises(RuntimeError):
            r.validate()

    @pytest.mark.parametrize("filename", ["sort.json", "sort.py"])
    def test_array(self, filename):
        """Can we fetch PDAL data as a numpy array"""
        r = get_pipeline(filename)
        r.execute()
        arrays = r.arrays
        assert len(arrays) == 1

        a = arrays[0]
        assert a[0][0] == 635619.85
        assert a[1064][2] == 456.92

    @pytest.mark.parametrize("filename", ["sort.json", "sort.py"])
    def test_metadata(self, filename):
        """Can we fetch PDAL metadata"""
        r = get_pipeline(filename)
        with pytest.raises(RuntimeError):
            r.metadata
        r.execute()
        j = json.loads(r.metadata)
        assert j["metadata"]["readers.las"][0]["count"] == 1065

    @pytest.mark.parametrize("filename", ["sort.json", "sort.py"])
    def test_schema(self, filename):
        """Fetching a schema works"""
        r = get_pipeline(filename)
        with pytest.raises(RuntimeError):
            r.schema
        r.execute()
        assert r.schema["schema"]["dimensions"][0]["name"] == "X"

    @pytest.mark.parametrize("filename", ["sort.json", "sort.py"])
    def test_no_execute(self, filename):
        """Does fetching arrays without executing throw an exception"""
        r = get_pipeline(filename)
        with pytest.raises(RuntimeError):
            r.arrays

    @pytest.mark.parametrize("filename", ["chip.json", "chip.py"])
    def test_merged_arrays(self, filename):
        """Can we fetch multiple point views from merged PDAL data"""
        r = get_pipeline(filename)
        r.execute()
        arrays = r.arrays
        assert len(arrays) == 43

    @pytest.mark.parametrize("filename", ["reproject.json", "reproject.py"])
    def test_logging(self, filename):
        """Can we fetch log output"""
        r = get_pipeline(filename)
        count = r.execute()
        assert count == 789
        # assert r.log.split()[0] == "(pypipeline"


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
    @pytest.mark.parametrize("filename", ["sort.json", "sort.py"])
    def test_no_execute(self, filename):
        """Does fetching meshes without executing throw an exception"""
        r = get_pipeline(filename)
        with pytest.raises(RuntimeError):
            r.meshes

    @pytest.mark.parametrize("filename", ["mesh.json", "mesh.py"])
    def test_mesh(self, filename):
        """Can we fetch PDAL face data as a numpy array"""
        r = get_pipeline(filename)
        points = r.execute()
        assert points == 1065
        meshes = r.meshes
        assert len(meshes) == 24

        m = meshes[0]
        assert str(m.dtype) == "[('A', '<u4'), ('B', '<u4'), ('C', '<u4')]"
        assert len(m) == 134
        assert m[0][0] == 29
