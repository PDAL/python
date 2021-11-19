from skbuild import setup

# Get the version from the pdal module
with open("pdal/__init__.py", "r") as fp:
    for line in fp:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip("\"'")
            break
    else:
        raise ValueError("Could not determine Python package version")

with open("README.rst", "r", encoding="utf-8") as fp:
    readme = fp.read()

with open("CHANGES.txt", "r", encoding="utf-8") as fp:
    changes = fp.read()

setup(
    name="PDAL",
    version=version,
    description="Point cloud data processing",
    license="BSD",
    keywords="point cloud spatial",
    author="Howard Butler",
    author_email="howard@hobu.co",
    maintainer="Howard Butler",
    maintainer_email="howard@hobu.co",
    url="https://pdal.io",
    long_description=readme + "\n\n" + changes,
    long_description_content_type="text/x-rst",
    packages=["pdal"],
    install_requires=["numpy"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: GIS",
    ],
)
