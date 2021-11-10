import json
import logging
import os
import sys

import numpy as np
import pytest

import pdal

DATADIRECTORY = os.path.join(os.path.dirname(__file__), "data")


def get_pipeline(filename):
    with open(os.path.join(DATADIRECTORY, filename), "r") as f:
        if filename.endswith(".json"):
            pipeline = pdal.Pipeline(f.read())
        elif filename.endswith(".py"):
            pipeline = eval(f.read(), vars(pdal))
    return pipeline


def test_dimensions():
    """Ask PDAL for its valid dimensions list"""
    dims = pdal.dimensions
    assert 71 < len(dims) < 120


class TestPipeline:
    @pytest.mark.parametrize("filename", ["sort.json", "sort.py"])
    def test_construction(self, filename):
        """Can we construct a PDAL pipeline"""
        assert isinstance(get_pipeline(filename), pdal.Pipeline)

        # construct Pipeline from a sequence of stages
        r = pdal.Reader("r")
        f = pdal.Filter("f")
        for spec in (r, f), [r, f]:
            p = pdal.Pipeline(spec)
            assert isinstance(p, pdal.Pipeline)
            assert len(p.stages) == 2

    @pytest.mark.parametrize(
        "pipeline",
        [
            "{}",
            '{"foo": []}',
            "[1, 2]",
            '{"pipeline": [["a.las", "b.las"], "c.las"]}',
        ],
    )
    def test_invalid_json(self, pipeline):
        """Do we complain with bad pipelines"""
        json.loads(pipeline)
        with pytest.raises(ValueError):
            pdal.Pipeline(pipeline)

    @pytest.mark.parametrize("filename", ["sort.json", "sort.py"])
    def test_execution(self, filename):
        """Can we execute a PDAL pipeline"""
        r = get_pipeline(filename)
        r.execute()
        assert len(r.pipeline) > 200

    @pytest.mark.parametrize("filename", ["bad.json", "bad.py"])
    def test_validate(self, filename):
        """Do we complain with bad pipelines"""
        r = get_pipeline(filename)
        with pytest.raises(RuntimeError):
            r.execute()

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
        assert j["metadata"]["readers.las"]["count"] == 1065

    @pytest.mark.parametrize("filename", ["sort.json", "sort.py"])
    def test_schema(self, filename):
        """Fetching a schema works"""
        r = get_pipeline(filename)
        with pytest.raises(RuntimeError):
            r.schema

        r.execute()
        assert r.schema["schema"]["dimensions"][0]["name"] == "X"

    @pytest.mark.parametrize("filename", ["sort.json", "sort.py"])
    def test_pipeline(self, filename):
        """Can we fetch PDAL pipeline string"""
        r = get_pipeline(filename)
        with pytest.raises(RuntimeError):
            r.pipeline

        r.execute()
        assert json.loads(r.pipeline) == {
            "pipeline": [
                {
                    # TODO: update this after https://github.com/PDAL/PDAL/issues/3574
                    "filename": f"test/data{os.sep}1.2-with-color.las",
                    "tag": "readers_las1",
                    "type": "readers.las",
                },
                {
                    "dimension": "X",
                    "inputs": ["readers_las1"],
                    "tag": "filters_sort1",
                    "type": "filters.sort",
                },
            ]
        }

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

    @pytest.mark.parametrize("filename", ["chip.json", "chip.py"])
    def test_stages(self, filename):
        """Can we break up a pipeline as a sequence of stages"""
        stages = pdal.Reader("test/data/autzen-utm.las").pipeline().stages
        assert len(stages) == 1

        stages = get_pipeline(filename).stages
        assert len(stages) == 3

        assert isinstance(stages[0], pdal.Reader)
        assert stages[0].type == "readers.las"

        assert isinstance(stages[1], pdal.Filter)
        assert stages[1].type == "filters.chipper"

        assert isinstance(stages[2], pdal.Writer)
        assert stages[2].type == "writers.las"

    def test_pipe_stages(self):
        """Can we build a pipeline by piping stages together"""
        read = pdal.Reader("test/data/autzen-utm.las")
        frange = pdal.Filter.range(limits="Intensity[50:200)")
        fsplitter = pdal.Filter.splitter(length=1000)
        fdelaunay = pdal.Filter.delaunay(inputs=[frange, fsplitter])

        # pipe stages together
        pipeline = read | frange | fsplitter | fdelaunay
        pipeline.execute()

        # pipe a pipeline to a stage
        pipeline = read | (frange | fsplitter | fdelaunay)
        pipeline.execute()

        # pipe a pipeline to a pipeline
        pipeline = (read | frange) | (fsplitter | fdelaunay)
        pipeline.execute()

    def test_pipe_stage_errors(self):
        """Do we complain with piping invalid objects"""
        r = pdal.Reader("r", tag="r")
        f = pdal.Filter("f")
        w = pdal.Writer("w", inputs=["r", f])

        with pytest.raises(TypeError):
            r | (f, w)
        with pytest.raises(TypeError):
            (r, f) | w
        with pytest.raises(TypeError):
            (r, f) | (f, w)

        pipeline = r | w
        with pytest.raises(RuntimeError) as ctx:
            pipeline.execute()
        assert "Undefined stage 'f'" in str(ctx.value)

    def test_inputs(self):
        """Can we combine pipelines with inputs"""
        data = np.load(os.path.join(DATADIRECTORY, "test3d.npy"))
        f = pdal.Filter.splitter(length=1000)
        pipeline = f.pipeline(data)
        pipeline.execute()

        # a pipeline with inputs can be followed by stage/pipeline
        (pipeline | pdal.Writer.null()).execute()
        (pipeline | (f | pdal.Writer.null())).execute()

        # a pipeline with inputs cannot follow another stage/pipeline
        with pytest.raises(ValueError):
            pdal.Reader("r") | pipeline
        with pytest.raises(ValueError):
            (pdal.Reader("r") | f) | pipeline

    def test_infer_stage_type(self):
        """Can we infer stage type from the filename"""
        assert pdal.Reader("foo.las").type == "readers.las"
        assert pdal.Writer("foo.las").type == "writers.las"
        assert pdal.Reader("foo.xxx").type == ""
        assert pdal.Writer("foo.xxx").type == ""
        assert pdal.Reader().type == ""
        assert pdal.Writer().type == ""

    # fails against PDAL master; see https://github.com/PDAL/PDAL/issues/3566
    @pytest.mark.xfail
    @pytest.mark.parametrize("filename", ["reproject.json", "reproject.py"])
    def test_logging(self, filename):
        """Can we fetch log output"""
        r = get_pipeline(filename)
        assert r.loglevel == logging.ERROR
        assert r.log == ""

        for loglevel in logging.CRITICAL, -1:
            with pytest.raises(ValueError):
                r.loglevel = loglevel

        count = r.execute()
        assert count == 789
        assert r.log == "entered filter()\n" + "exiting filter()\n"

        r.loglevel = logging.DEBUG
        assert r.loglevel == logging.DEBUG
        count = r.execute()
        assert count == 789
        assert "(pypipeline readers.las Debug)" in r.log
        assert "(pypipeline filters.python Debug)" in r.log
        assert "\nentered filter()\n" in r.log
        assert "\nexiting filter()\n" in r.log
        assert "(pypipeline writers.las Debug)" in r.log

    def test_only_readers(self):
        """Does a pipeline that consists of only readers return the merged data"""
        read = pdal.Reader("test/data/*.las")
        r1 = read.pipeline()
        count1 = r1.execute()
        array1 = r1.arrays[0]

        r2 = read | read
        count2 = r2.execute()
        array2 = r2.arrays[0]

        assert count2 == 2 * count1
        np.testing.assert_array_equal(np.concatenate([array1, array1]), array2)


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

    @pytest.mark.parametrize("filename", ["mesh.json", "mesh.py"])
    def test_meshio(self, filename):
        r = get_pipeline(filename)
        r.execute()
        mesh = r.get_meshio(0)
        triangles = mesh.cells_dict["triangle"]
        assert len(triangles) == 134
        assert triangles[0][0] == 29
