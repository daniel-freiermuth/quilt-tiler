# oesu-tiler

Converts decrypted o-charts OESU files (SENC v201) into a PMTiles vector tile archive

## Prerequisites

**Decrypted `.oesu` files** — use
[oesenc-export](../oesenc-export/README.md) to export charts from OpenCPN's
local plugin cache. The exported files land in `oesenc-export/exported/`.

## Usage

```
oesu2geojson -o chart.pmtiles *.oesu
```

Beware, this can consume significant amounts of RAM.

## Known limitations

- **Tile size**: area and line features are written to every tile their bounding
  box overlaps, without clipping to tile boundaries. The MVT spec permits
  out-of-extent coordinates; MapLibre clips client-side. Result: ~15 GB output
  for 582 charts (~35 KB/tile average). Tile boundary clipping would reduce
  this ~4–6×.
- **Memory**: all tile bytes are buffered in a `BTreeMap` before writing.
  A future streaming pass would reduce peak RSS.
- **Raster charts** (`.oernc`, BSB format) are detected and rejected with a
  clear error message. Only vector SENC v201 is supported.
