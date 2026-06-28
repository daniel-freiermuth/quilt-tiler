# Quilt-tiler

Quilts cell-based charts like traditional seacharts with a few meaningful mixed zoom levels into a single tile layer.

Supported input formats: S57-ish (.000 (planned), decrypted oesu, osenc, GeoJSON-representations),
and `.rnc` raster cells.

Supported output formats: pmtiles carrying mvt (vector charts) or png (raster charts; mbtiles/mlt planned)
and an accompanying style.json + Signal K metadata.json.

## Usage

```
quilt-tiler -o chart.pmtiles <input-charts>          # vector: .oesu/.osenc cells
quilt-tiler -o chart.pmtiles <cells>/*.rnc            # raster: rnc cells
