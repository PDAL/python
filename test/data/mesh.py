Reader("test/data/1.2-with-color.las") | Filter.splitter(length=1000) | Filter.delaunay()
