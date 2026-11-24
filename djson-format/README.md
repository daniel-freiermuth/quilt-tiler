# djson-format

Parser for `.djson` vector cells: S-57 object classes as JSON layers, geometry
as flat spherical-Mercator coordinate arrays split by `_SL` / `_SO` / `_SP`.

Produces typed records with `WGS-84` [`geo`](https://docs.rs/geo) geometry, using
`rnc-format`'s Mercator projection (both formats share the same sphere).

Deliberately excludes HTTP, decompression and output encoding: consumers fetch
and decompress the bytes, then turn records into GeoJSON, vector tiles or
`s57::S57Cell`s.
