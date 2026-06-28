# rnc-format

Shared parser for raster-cell `.rnc` files: binary layout, footer schema
(including the reverse-engineered `cover` polygon), and the format's own
Mercator projection.

Used by:
- `quilt-tiler` (this repo, workspace root) — batch-converts `.rnc` cells to
  `PMTiles`.

Split out after the same wire format was independently — and divergently —
reimplemented three times across those two projects (different interpolation
spaces, different `cover` schema assumptions, no shared test coverage). See
each crate's `src/lib.rs` doc comment for the format layout and the
coordinate-system caveats.

Deliberately excludes PNG decoding/rendering: each consumer has different
needs there (on-demand decode + LRU eviction for a live tile server vs. bulk
decode + disjoint-coverage compositing for batch tiling) — only the format
itself is shared.
