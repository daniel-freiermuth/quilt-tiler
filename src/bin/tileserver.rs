//! Live tile server for decrypted `OESU`/`OSENC` vector charts or raster
//! `.rnc` cells.
//!
//! Parses all input cells once at startup, then serves `MapLibre` GL tiles
//! and a matching `style.json` on demand — same quilting/coverage logic as
//! the batch `quilt-tiler` binary ([`quilt_tiler::tiles::render_tile`]), just
//! computed per request instead of written to a `PMTiles` archive.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::Router;
use axum::extract::{Path as TilePath, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use bytes::Bytes;
use clap::Parser;
use moka::sync::Cache;
use quilt_tiler::{loader, rnc, style, tiles};
use tower_http::cors::CorsLayer;
use tracing::{error, info};

/// Serve decrypted `OESU`/`OSENC` vector charts or raster `.rnc` cells as
/// `MapLibre` GL tiles and a `style.json`, generated on the fly.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Input chart files to serve (glob or explicit paths): decrypted
    /// `.oesu`/`.osenc` vector cells, or raster `.rnc` cells.
    /// All inputs in one run must be the same kind — pick one source type
    /// per server, same constraint as the batch `quilt-tiler` binary.
    #[arg(required = true)]
    input: Vec<PathBuf>,

    /// Address to bind the HTTP server to.
    #[arg(long, default_value = "0.0.0.0:3000")]
    bind: SocketAddr,

    /// Public base URL (`scheme://host[:port]`, no trailing slash) embedded
    /// in `/style.json`'s tile source. Defaults to deriving
    /// `http://<request's Host header>` per request, which works for
    /// local/LAN access with zero configuration but not behind an
    /// HTTPS-terminating reverse proxy.
    #[arg(long)]
    public_url: Option<String>,

    /// Vector input only. Depth (metres) at or above which water is
    /// considered dangerous — see `quilt-tiler --help` for the full
    /// explanation, identical semantics here.
    #[arg(long, default_value_t = 3.0)]
    safety_depth: f64,

    /// Vector input only. Upper boundary of the "shallow but navigable"
    /// zone (metres).
    #[arg(long, default_value_t = 10.0)]
    shoal_depth: f64,

    /// Cap served tiles at this zoom level.
    #[arg(long)]
    max_zoom: Option<u8>,

    /// Shift every chart's native zoom level by this amount (positive =
    /// finer, negative = coarser).
    #[arg(long, default_value_t = 0.0)]
    zoom_offset: f64,

    /// In-memory tile cache size in megabytes, weighed by each tile's
    /// encoded byte size (so capacity is approximate memory use, not entry
    /// count). Concurrent requests for the same uncached tile are
    /// coalesced — only one render runs, the rest wait for its result.
    /// `0` disables the cache.
    #[arg(long, default_value_t = 256)]
    cache_mb: u64,
}

/// The parsed cells this server quilts tiles from, plus the zoom range
/// [`quilt_tiler::tiles::zoom_range_and_bounds`] computed for them once at
/// startup — recomputing it per request would be pure overhead since the
/// cell set never changes for the life of the process.
enum Sources {
    Vector {
        cells: Vec<s57::S57Cell>,
        min_zoom: u8,
        max_zoom: u8,
    },
    Raster {
        cells: Vec<rnc::RncCell>,
        min_zoom: u8,
        max_zoom: u8,
    },
}

/// `(z, x, y)` → rendered tile bytes, or `None` for a tile nothing covers.
/// Caching `None` too matters: a marine chart's open-water tiles repeat the
/// same empty result over huge stretches of panning, and that's still real
/// candidate-filtering work across every loaded cell without this.
type TileCache = Cache<(u8, u32, u32), Option<Bytes>>;

/// Weigh a cached entry by its tile size in bytes (a `None` "no coverage"
/// entry counts as a small fixed weight) — bounds the cache by approximate
/// memory use rather than a fixed entry count, since tile sizes vary from a
/// few hundred bytes to several hundred KB.
#[allow(clippy::cast_possible_truncation)] // tiles are well under u32::MAX bytes
#[allow(clippy::ref_option)] // signature fixed by moka's `weigher` closure type
fn tile_weight(_key: &(u8, u32, u32), value: &Option<Bytes>) -> u32 {
    value.as_ref().map_or(64, |bytes| bytes.len() as u32 + 64)
}

struct AppState {
    sources: Sources,
    cache: Option<TileCache>,
    public_url: Option<String>,
    safety_depth: f64,
    shoal_depth: f64,
    zoom_offset: f64,
}

impl AppState {
    /// MIME type for this server's `/tiles` responses — constant for the
    /// life of the process, one source kind per server.
    const fn content_type(&self) -> &'static str {
        match &self.sources {
            Sources::Vector { .. } => "application/x-protobuf",
            Sources::Raster { .. } => "image/png",
        }
    }

    /// Render one `(z, x, y)` tile, dispatching to the loaded source's
    /// [`TileSource`] impl. Bypasses the cache.
    fn render_uncached(&self, z: u8, x: u32, y: u32) -> Result<Option<Bytes>> {
        match &self.sources {
            Sources::Vector { cells, .. } => {
                Ok(tiles::render_tile(cells, z, x, y, self.zoom_offset)?.map(Bytes::from))
            }
            Sources::Raster { cells, .. } => {
                Ok(tiles::render_tile(cells, z, x, y, self.zoom_offset)?.map(Bytes::from))
            }
        }
    }

    /// Render `(z, x, y)`, transparently caching the result (including a
    /// "nothing covers this tile" miss) when a cache is configured.
    ///
    /// Concurrent requests for the same uncached tile are coalesced by
    /// [`moka::sync::Cache::try_get_with`]: only one render runs, the rest
    /// wait for its result instead of duplicating the work.
    fn render(&self, z: u8, x: u32, y: u32) -> Result<Option<(Bytes, &'static str)>> {
        let content_type = self.content_type();
        let tile = match &self.cache {
            Some(cache) => cache
                .try_get_with((z, x, y), || self.render_uncached(z, x, y))
                .map_err(|e| anyhow::anyhow!("{e}"))?,
            None => self.render_uncached(z, x, y)?,
        };
        Ok(tile.map(|bytes| (bytes, content_type)))
    }

    /// Build the `style.json` body, embedding `tile_url` as the source's
    /// tile template.
    fn style_json(&self, tile_url: &str) -> String {
        match &self.sources {
            Sources::Vector {
                min_zoom, max_zoom, ..
            } => style::build_style(
                self.safety_depth,
                self.shoal_depth,
                tile_url,
                *min_zoom,
                *max_zoom,
            ),
            Sources::Raster {
                min_zoom, max_zoom, ..
            } => style::build_raster_style(tile_url, *min_zoom, *max_zoom),
        }
    }

    /// Base URL for tile templates: the configured `--public-url`, or
    /// `http://<Host header>` derived from the live request.
    fn base_url(&self, headers: &HeaderMap) -> String {
        self.public_url.clone().unwrap_or_else(|| {
            let host = headers
                .get(header::HOST)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("localhost:3000");
            format!("http://{host}")
        })
    }
}

async fn style_handler(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let tile_url = format!("{}/tiles/{{z}}/{{x}}/{{y}}", state.base_url(&headers));
    let body = state.style_json(&tile_url);
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        body,
    )
        .into_response()
}

async fn tile_handler(
    State(state): State<Arc<AppState>>,
    TilePath((z, x, y)): TilePath<(u8, u32, u32)>,
) -> Response {
    let result = tokio::task::spawn_blocking(move || state.render(z, x, y)).await;
    match result {
        Ok(Ok(Some((bytes, content_type)))) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, content_type)],
            bytes,
        )
            .into_response(),
        Ok(Ok(None)) => StatusCode::NO_CONTENT.into_response(),
        Ok(Err(e)) => {
            error!(error = %e, z, x, y, "tile render failed");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
        Err(e) => {
            error!(error = %e, z, x, y, "tile render task panicked");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();

    // Same one-kind-per-run constraint as the batch CLI's PMTiles output —
    // here it's not a format limitation, just keeps one server's `/tiles`
    // route unambiguous (MVT vs PNG, one content type).
    let raster_count = args.input.iter().filter(|p| loader::is_rnc(p)).count();
    let sources = match raster_count {
        0 => {
            let cells = loader::load_s57_cells(&args.input, args.zoom_offset);
            anyhow::ensure!(!cells.is_empty(), "no vector cells parsed from input");
            let (min_zoom, max_zoom, _bounds) =
                tiles::zoom_range_and_bounds(&cells, args.max_zoom, args.zoom_offset)?;
            info!(
                cells = cells.len(),
                min_zoom, max_zoom, "vector cells loaded"
            );
            Sources::Vector {
                cells,
                min_zoom,
                max_zoom,
            }
        }
        n if n == args.input.len() => {
            let cells = loader::load_rnc_cells(&args.input, args.zoom_offset);
            anyhow::ensure!(!cells.is_empty(), "no raster cells parsed from input");
            let (min_zoom, max_zoom, _bounds) =
                tiles::zoom_range_and_bounds(&cells, args.max_zoom, args.zoom_offset)?;
            info!(
                cells = cells.len(),
                min_zoom, max_zoom, "raster cells loaded"
            );
            Sources::Raster {
                cells,
                min_zoom,
                max_zoom,
            }
        }
        n => anyhow::bail!(
            "cannot mix raster (.rnc) and vector inputs in one run ({n} of {} inputs are .rnc); serve them separately",
            args.input.len()
        ),
    };

    let cache = (args.cache_mb > 0).then(|| {
        Cache::builder()
            .max_capacity(args.cache_mb * 1024 * 1024)
            .weigher(tile_weight)
            .build()
    });
    info!(
        cache_mb = args.cache_mb,
        cache_enabled = cache.is_some(),
        "tile cache configured"
    );

    let state = Arc::new(AppState {
        sources,
        cache,
        public_url: args.public_url,
        safety_depth: args.safety_depth,
        shoal_depth: args.shoal_depth,
        zoom_offset: args.zoom_offset,
    });

    let app = Router::new()
        .route("/style.json", get(style_handler))
        .route("/tiles/{z}/{x}/{y}", get(tile_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);

    info!(bind = %args.bind, "starting tile server");
    let listener = tokio::net::TcpListener::bind(args.bind)
        .await
        .with_context(|| format!("binding {}", args.bind))?;
    axum::serve(listener, app).await.context("serving")?;
    Ok(())
}
