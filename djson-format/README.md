# djson-format

Parser for `.djson` vector cells: S-57 object classes as JSON layers, geometry
as flat spherical-Mercator coordinate arrays split by `_SL` / `_SO` / `_SP`.

- `detect` recognises a document by its content.
- `parse` produces typed records with `WGS-84` [`geo`](https://docs.rs/geo)
  geometry, using `rnc-format`'s Mercator projection (both formats share the
  same sphere).
- `S57Cell::try_from(cell)` converts a cell into an `s57::S57Cell`, typing
  attributes by the S-57 catalogue.

Deliberately excludes HTTP and decompression: consumers fetch and decompress
the bytes.
