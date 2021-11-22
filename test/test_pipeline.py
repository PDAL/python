import json
import logging
import os
import sys

import numpy as np
import pytest

import pdal

DATADIRECTORY = os.path.join(os.path.dirname(__file__), "data")


def a_filter(ins, outs):
    return True


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
    def test_execute(self, filename):
        """Can we execute a PDAL pipeline"""
        r = get_pipeline(filename)
        count = r.execute()
        assert count == 1065

    @pytest.mark.parametrize("filename", ["range.json", "range.py"])
    def test_execute_streaming(self, filename):
        r = get_pipeline(filename)
        assert r.streamable
        count = r.execute()
        count2 = r.execute_streaming(chunk_size=100)
        assert count == count2

    @pytest.mark.parametrize("filename", ["sort.json", "sort.py"])
    def test_execute_streaming_non_streamable(self, filename):
        r = get_pipeline(filename)
        assert not r.streamable
        with pytest.raises(RuntimeError) as info:
            r.execute_streaming()
        assert "Attempting to use stream mode" in str(info.value)

    @pytest.mark.parametrize("filename", ["bad.json", "bad.py"])
    def test_validate(self, filename):
        """Do we complain with bad pipelines"""
        r = get_pipeline(filename)
        with pytest.raises(RuntimeError) as info:
            r.execute()
        if os.name == "nt":
            assert "Unable to open stream for" in str(info.value)
        else:
            assert "No such file or directory" in str(info.value)

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
        with pytest.raises(RuntimeError) as info:
            r.metadata
        assert "Pipeline has not been executed" in str(info.value)

        r.execute()
        j = json.loads(r.metadata)
        assert j["metadata"]["readers.las"]["count"] == 1065

    @pytest.mark.parametrize("filename", ["sort.json", "sort.py"])
    def test_schema(self, filename):
        """Fetching a schema works"""
        r = get_pipeline(filename)
        with pytest.raises(RuntimeError) as info:
            r.schema
        assert "Pipeline has not been executed" in str(info.value)

        r.execute()
        assert r.schema["schema"]["dimensions"][0]["name"] == "X"

    @pytest.mark.parametrize("filename", ["sort.json", "sort.py"])
    def test_pipeline(self, filename):
        """Can we fetch PDAL pipeline string"""
        r = get_pipeline(filename)
        with pytest.raises(RuntimeError) as info:
            r.pipeline
        assert "Pipeline has not been executed" in str(info.value)

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
        with pytest.raises(RuntimeError) as info:
            r.arrays
        assert "Pipeline has not been executed" in str(info.value)

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
        with pytest.raises(RuntimeError) as info:
            pipeline.execute()
        assert "Undefined stage 'f'" in str(info.value)

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

    def test_streamable(self):
        """Can we distinguish streamable from non-streamable stages and pipeline"""
        rs = pdal.Reader(type="readers.las", filename="foo")
        assert rs.streamable is True
        assert pdal.Reader.las("foo").streamable is True
        assert pdal.Reader("foo.las").streamable is True

        rn = pdal.Reader(type="readers.pts", filename="foo")
        assert rn.streamable is False
        assert pdal.Reader.pts("foo").streamable is False
        assert pdal.Reader("foo.pts").streamable is False

        fs = pdal.Filter(type="filters.crop")
        assert fs.streamable is True
        assert pdal.Filter.crop().streamable is True

        fn = pdal.Filter(type="filters.cluster")
        assert fn.streamable is False
        assert pdal.Filter.cluster().streamable is False

        ws = pdal.Writer(type="writers.ogr", filename="foo")
        assert ws.streamable is True
        assert pdal.Writer.ogr(filename="foo").streamable is True
        assert pdal.Writer("foo.shp").streamable is True

        wn = pdal.Writer(type="writers.glb", filename="foo")
        assert wn.streamable is False
        assert pdal.Writer.gltf("foo").streamable is False
        assert pdal.Writer("foo.glb").streamable is False

        assert (rs | fs | ws).streamable is True
        assert (rn | fs | ws).streamable is False
        assert (rs | fn | ws).streamable is False
        assert (rs | fs | wn).streamable is False

    @pytest.mark.parametrize("filename", ["chip.json", "chip.py"])
    def test_logging(self, filename):
        """Can we fetch log output"""
        r = get_pipeline(filename)
        assert r.loglevel == logging.ERROR
        assert r.log == ""

        for loglevel in logging.CRITICAL, -1:
            with pytest.raises(ValueError):
                r.loglevel = loglevel

        count = r.execute()
        assert count == 1065
        assert r.log == ""

        r.loglevel = logging.DEBUG
        assert r.loglevel == logging.DEBUG
        count = r.execute()
        assert count == 1065
        assert "(pypipeline readers.las Debug)" in r.log
        assert "(pypipeline Debug) Executing pipeline in standard mode" in r.log
        assert "(pypipeline writers.las Debug)" in r.log

    @pytest.mark.skipif(
        not hasattr(pdal.Filter, "python"),
        reason="filters.python PDAL plugin is not available",
    )
    @pytest.mark.parametrize("filename", ["reproject.json", "reproject.py"])
    def test_logging_filters_python(self, filename):
        """Can we fetch log output including print() statements from filters.python"""
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

    @pytest.mark.skipif(
        not hasattr(pdal.Filter, "python"),
        reason="filters.python PDAL plugin is not available",
    )
    def test_filters_python(self):
        r = pdal.Reader("test/data/autzen-utm.las")
        f = pdal.Filter.python(script=__file__, function="a_filter", module="anything")
        count = (r | f).execute()
        assert count == 1065

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
            dtype=[("X", float), ("Y", float), ("Z", float)],
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
            dtype=[("X", float), ("Y", float), ("Z", float)],
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
        with pytest.raises(RuntimeError) as info:
            r.meshes
        assert "Pipeline has not been executed" in str(info.value)

    @pytest.mark.parametrize("filename", ["mesh.json", "mesh.py"])
    def test_mesh(self, filename):
        """Can we fetch PDAL face data as a numpy array"""
        r = get_pipeline(filename)
        r.execute()
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


class TestPipelineIterator:
    @pytest.mark.parametrize("filename", ["sort.json", "sort.py"])
    def test_non_streamable(self, filename):
        r = get_pipeline(filename)
        assert not r.streamable
        with pytest.raises(RuntimeError) as info:
            next(r.iterator(chunk_size=100))
        assert "Attempting to use stream mode" in str(info.value)

    @pytest.mark.parametrize("filename", ["range.json", "range.py"])
    def test_array(self, filename):
        """Can we fetch PDAL data as numpy arrays"""
        r = get_pipeline(filename)
        count = r.execute()
        arrays = r.arrays
        assert len(arrays) == 1
        array = arrays[0]
        assert count == len(array)

        for _ in range(10):
            arrays = list(r.iterator(chunk_size=100))
            assert len(arrays) == 11
            concat_array = np.concatenate(arrays)
            np.testing.assert_array_equal(array, concat_array)

    @pytest.mark.parametrize("filename", ["range.json", "range.py"])
    def test_StopIteration(self, filename):
        """Is StopIteration raised when the iterator is exhausted"""
        r = get_pipeline(filename)
        it = r.iterator(chunk_size=100)
        for array in it:
            assert isinstance(array, np.ndarray)
        with pytest.raises(StopIteration):
            next(it)
        assert next(it, None) is None

    @pytest.mark.parametrize("filename", ["range.json", "range.py"])
    def test_metadata(self, filename):
        """Can we fetch PDAL metadata"""
        r = get_pipeline(filename)
        r.execute()

        it = r.iterator(chunk_size=100)
        for _ in it:
            pass

        assert r.metadata == it.metadata

    @pytest.mark.parametrize("filename", ["range.json", "range.py"])
    def test_schema(self, filename):
        """Fetching a schema works"""
        r = get_pipeline(filename)
        r.execute()

        it = r.iterator(chunk_size=100)
        for _ in it:
            pass

        assert r.schema == it.schema

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
        non_streaming_array = np.concatenate(p.arrays)
        for chunk_size in range(5, 100, 5):
            streaming_arrays = list(p.iterator(chunk_size=chunk_size))
            np.testing.assert_array_equal(
                np.concatenate(streaming_arrays), non_streaming_array
            )

    @pytest.mark.parametrize("filename", ["range.json", "range.py"])
    def test_premature_exit(self, filename):
        """Can we stop iterating before all arrays are fetched"""
        r = get_pipeline(filename)
        r.execute()
        assert len(r.arrays) == 1
        array = r.arrays[0]

        for _ in range(10):
            for array2 in r.iterator(chunk_size=100):
                np.testing.assert_array_equal(array2, array[: len(array2)])
                break

    @pytest.mark.parametrize("filename", ["range.json", "range.py"])
    def test_multiple_iterators(self, filename):
        """Can we create multiple independent iterators"""
        r = get_pipeline(filename)
        it1 = r.iterator(chunk_size=100)
        it2 = r.iterator(chunk_size=100)
        for a1, a2 in zip(it1, it2):
            np.testing.assert_array_equal(a1, a2)
        assert next(it1, None) is None
        assert next(it2, None) is None
