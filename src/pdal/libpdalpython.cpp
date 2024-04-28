#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/stl/filesystem.h>
#include <iostream>

#include <pdal/pdal_config.hpp>
#include <pdal/StageFactory.hpp>

#define NPY_TARGET_VERSION NPY_1_22_API_VERSION
#define NPY_NO_DEPRECATED_API NPY_1_22_API_VERSION

#define PY_ARRAY_UNIQUE_SYMBOL PDAL_ARRAY_API

#include <numpy/arrayobject.h>

#include "PyArray.hpp"
#include "PyDimension.hpp"
#include "PyPipeline.hpp"
#include "StreamableExecutor.hpp"

namespace py = pybind11;

namespace pdal {
    using namespace py::literals;

    py::object getInfo() {
        return py::module_::import("types").attr("SimpleNamespace")(
                "version"_a = pdal::Config::versionString(),
                "major"_a = pdal::Config::versionMajor(),
                "minor"_a = pdal::Config::versionMinor(),
                "patch"_a = pdal::Config::versionPatch(),
                "debug"_a = pdal::Config::debugInformation(),
                "sha1"_a = pdal::Config::sha1(),
                "plugin"_a = pdal::Config::pluginInstallPath()
        );
    };

   std::vector<py::dict> getDrivers() {
        std::vector<py::dict> drivers;

        pdal::StageFactory f(false);
        pdal::PluginManager<pdal::Stage>::loadAll();
        pdal::StringList stages = pdal::PluginManager<pdal::Stage>::names();

        pdal::StageExtensions& extensions = pdal::PluginManager<pdal::Stage>::extensions();
        for (auto name : stages)
        {
            pdal::Stage *s = f.createStage(name);
            std::string description = pdal::PluginManager<Stage>::description(name);
            std::string link = pdal::PluginManager<Stage>::link(name);
            std::vector<std::string> extension_names = extensions.extensions(name);

            py::dict d(
                "name"_a=name,
                "description"_a=description,
                "streamable"_a=s->pipelineStreamable(),
                "extensions"_a=extension_names
            );
            f.destroyStage(s);
            drivers.push_back(std::move(d));
        }
        return drivers;

    };

   py::object getOptions() {
        py::object json = py::module_::import("json");
        py::dict stageOptions;

        pdal::StageFactory f;
        pdal::PluginManager<pdal::Stage>::loadAll();
        pdal::StringList stages = pdal::PluginManager<pdal::Stage>::names();

        for (auto name : stages)
        {
            pdal::Stage *s = f.createStage(name);
            pdal::ProgramArgs args;
            s->addAllArgs(args);
            std::ostringstream ostr;
            args.dump3(ostr);
            py::str pystring(ostr.str());
            pystring.attr("strip");

            py::object j;

            try {
                j = json.attr("loads")(pystring);
            } catch (py::error_already_set &e) {
                std::cerr << "failed:" << name << "'" << ostr.str() << "'" <<std::endl;
                continue; // skip this one because we can't parse it
            }

            f.destroyStage(s);
            stageOptions[pybind11::cast(name)] = std::move(j);
       }
        return stageOptions;

    };

    std::vector<py::dict> getDimensions() {
        py::object np = py::module_::import("numpy");
        py::object dtype = np.attr("dtype");
        std::vector<py::dict> dims;
        for (const auto& dim: getValidDimensions())
        {
            py::dict d(
                "name"_a=dim.name,
                "description"_a=dim.description,
                "dtype"_a=dtype(dim.type + std::to_string(dim.size))
            );
            dims.push_back(std::move(d));
        }
        return dims;
    };

    std::string getReaderDriver(std::filesystem::path const& p)
    {
        return StageFactory::inferReaderDriver(p.string());
    }

    std::string getWriterDriver(std::filesystem::path const& p)
    {
        return StageFactory::inferWriterDriver(p.string());
    }

    using pdal::python::PipelineExecutor;
    using pdal::python::StreamableExecutor;

    class PipelineIterator : public StreamableExecutor {
    public:
        using StreamableExecutor::StreamableExecutor;

        py::object getSchema() {
            return py::module_::import("json").attr("loads")(StreamableExecutor::getSchema());
        }

        py::array executeNext() {
            PyArrayObject* arr(StreamableExecutor::executeNext());
            if (!arr)
                throw py::stop_iteration();

            return py::reinterpret_steal<py::array>((PyObject*)arr);
        }

        py::object getMetadata() {
            py::object json = py::module_::import("json");

            std::stringstream strm;
            MetadataNode root = (StreamableExecutor::getMetadata()).clone("metadata");
            pdal::Utils::toJSON(root, strm);


            py::bytes pybytes(strm.str());
            py::str pystring ( pybytes.attr("decode")("utf-8", "ignore"));

            py::object j;
            j = json.attr("loads")(pystring);

            return j;

        }

    };

    class Pipeline {
    public:
        point_count_t execute() {
            point_count_t response(0);
            {
                py::gil_scoped_release release;
                response = getExecutor()->execute();
            }
            return response;

        }

        point_count_t executeStream(point_count_t streamLimit) {
            point_count_t response(0);
            {
                py::gil_scoped_release release;
                response = getExecutor()->executeStream(streamLimit);
            }
            return response;
        }

        std::unique_ptr<PipelineIterator> iterator(int chunk_size, int prefetch) {
            return std::unique_ptr<PipelineIterator>(new PipelineIterator(
                getJson(), _inputs, _loglevel, chunk_size, prefetch
            ));
        }

        void setInputs(std::vector<py::array> ndarrays) {
            _inputs.clear();
            for (const auto& ndarray: ndarrays) {
                PyArrayObject* ndarray_ptr = (PyArrayObject*)ndarray.ptr();
                _inputs.push_back(std::make_shared<pdal::python::Array>(ndarray_ptr));
            }
            delExecutor();
        }

        int getLoglevel() { return _loglevel; }

        void setLogLevel(int level) { _loglevel = level; delExecutor(); }

        std::string getLog() { return getExecutor()->getLog(); }

        std::string getPipeline() { return getExecutor()->getPipeline(); }
        std::string getSrsWKT2() { return getExecutor()->getSrsWKT2(); }

        py::object getQuickInfo() {
            py::object json = py::module_::import("json");

            std::string response;
            {
                py::gil_scoped_release release;
                response = getExecutor()->getQuickInfo();
            }
            py::bytes pybytes(response);

            py::str pystring ( pybytes.attr("decode")("utf-8", "ignore"));
            pystring.attr("strip");

            py::object j;
            j = json.attr("loads")(pystring);

            return j;

        }

        py::object getMetadata() {
            py::object json = py::module_::import("json");

            py::bytes pybytes(getExecutor()->getMetadata());
            py::str pystring ( pybytes.attr("decode")("utf-8", "ignore"));

            py::object j;
            j = json.attr("loads")(pystring);

            return j;

        }

        py::object getSchema() {
            return py::module_::import("json").attr("loads")(getExecutor()->getSchema());
        }

        std::vector<py::array> getArrays() {
            std::vector<py::array> output;
            for (const auto &view: getExecutor()->views()) {
                PyArrayObject* arr(pdal::python::viewToNumpyArray(view));
                output.push_back(py::reinterpret_steal<py::array>((PyObject*)arr));
            }
            return output;
        }

        std::vector<py::array> getMeshes() {
            std::vector<py::array> output;
            for (const auto &view: getExecutor()->views()) {
                PyArrayObject* arr(pdal::python::meshToNumpyArray(view->mesh()));
                output.push_back(py::reinterpret_steal<py::array>((PyObject*)arr));
            }
            return output;
        }

        std::string getJson() const {
            PYBIND11_OVERRIDE_PURE_NAME(std::string, Pipeline, "toJSON", getJson);
        }

        bool hasInputs() { return !_inputs.empty(); }

        void copyInputs(const Pipeline& other) { _inputs = other._inputs; }

        void delExecutor() { _executor.reset(); }

        PipelineExecutor* getExecutor() {
            // We need to acquire the GIL before we create the executor
            // because this method does Python init stuff but pybind11 doesn't
            // automatically encapsulate it with a gil_scoped_acquire like it
            // does for all of the other methods it knows about
            py::gil_scoped_acquire acquire;
            if (!_executor)
                _executor.reset(new PipelineExecutor(getJson(), _inputs, _loglevel));
            return _executor.get();
        }

    private:
        std::unique_ptr<PipelineExecutor> _executor;
        std::vector<std::shared_ptr<pdal::python::Array>> _inputs;
        int _loglevel;
    };



    PYBIND11_MODULE(libpdalpython, m)
    {
        _import_array();

    py::class_<PipelineIterator>(m, "PipelineIterator")
        .def("__iter__", [](PipelineIterator &it) -> PipelineIterator& { return it; })
        .def("__next__", &PipelineIterator::executeNext)
        .def_property_readonly("log", &PipelineIterator::getLog)
        .def_property_readonly("schema", &PipelineIterator::getSchema)
        .def_property_readonly("srswkt2", &PipelineIterator::getSrsWKT2)
        .def_property_readonly("pipeline", &PipelineIterator::getPipeline)
        .def_property_readonly("metadata", &PipelineIterator::getMetadata);


    py::class_<Pipeline>(m, "Pipeline")
        .def(py::init<>())
        .def("execute", &Pipeline::execute)
        .def("execute_streaming", &Pipeline::executeStream, "chunk_size"_a=10000)
        .def("iterator", &Pipeline::iterator, "chunk_size"_a=10000, "prefetch"_a=0)
        .def_property("inputs", nullptr, &Pipeline::setInputs)
        .def_property("loglevel", &Pipeline::getLoglevel, &Pipeline::setLogLevel)
        .def_property_readonly("log", &Pipeline::getLog)
        .def_property_readonly("schema", &Pipeline::getSchema)
        .def_property_readonly("srswkt2", &Pipeline::getSrsWKT2)
        .def_property_readonly("pipeline", &Pipeline::getPipeline)
        .def_property_readonly("quickinfo", &Pipeline::getQuickInfo)
        .def_property_readonly("metadata", &Pipeline::getMetadata)
        .def_property_readonly("arrays", &Pipeline::getArrays)
        .def_property_readonly("meshes", &Pipeline::getMeshes)
        .def_property_readonly("_has_inputs", &Pipeline::hasInputs)
        .def("_copy_inputs", &Pipeline::copyInputs)
        .def("toJSON", &Pipeline::getJson)
        .def("_del_executor", &Pipeline::delExecutor);
    m.def("getInfo", &getInfo);
    m.def("getDrivers", &getDrivers);
    m.def("getOptions", &getOptions);
    m.def("getDimensions", &getDimensions);
    m.def("infer_reader_driver", &getReaderDriver);
    m.def("infer_writer_driver", &getWriterDriver);

    if (pdal::Config::versionMajor() < 2)
        throw pybind11::import_error("PDAL version must be >= 2.6");

    if (pdal::Config::versionMajor() == 2 && pdal::Config::versionMinor() < 6)
        throw pybind11::import_error("PDAL version must be >= 2.6");
    };

}; // namespace pdal
