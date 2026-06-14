# Quilt-tiler

Quilts cell-based charts like traditional seacharts with a few meaningful mixed zoom levels into a single tile layer.

Supported input formats: S57-ish (.000 (planned), decrypted oesu, osenc, GeoJSON-representations) and also raster-cells (planned).

Supported output formats: mbtiles/pmtiles carrying mvt or mlt (and soon png).

## Usage

```
quilt-tiler -o chart.pmtiles <input-charts>
```
