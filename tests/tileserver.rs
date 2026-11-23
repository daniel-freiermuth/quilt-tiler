//! End-to-end test of the `tileserver` binary: spawn it on fixture raster
//! charts and talk HTTP/1.1 to it over a plain TCP socket.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use common::{COARSE_BOX, COARSE_ZOOM, FINE_BOX, Scratch, tile_containing};

const READY_TIMEOUT: Duration = Duration::from_secs(20);
const IO_TIMEOUT: Duration = Duration::from_secs(10);

/// A running tileserver, killed on drop so a failing test never leaks it.
struct Server {
    child: Child,
    addr: SocketAddr,
}

impl Server {
    fn spawn(args: &[&std::ffi::OsStr]) -> Self {
        let addr = free_local_addr();
        let child = Command::new(env!("CARGO_BIN_EXE_tileserver"))
            .arg("--bind")
            .arg(addr.to_string())
            .args(args)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawning tileserver");
        let mut server = Self { child, addr };
        server.wait_ready();
        server
    }

    /// Poll-connect until the server accepts, failing fast if it exits.
    fn wait_ready(&mut self) {
        let deadline = Instant::now() + READY_TIMEOUT;
        loop {
            if let Some(status) = self.child.try_wait().expect("polling tileserver") {
                panic!("tileserver exited before accepting connections: {status}");
            }
            if TcpStream::connect_timeout(&self.addr, Duration::from_millis(200)).is_ok() {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "tileserver not accepting on {} after {READY_TIMEOUT:?}",
                self.addr
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    fn get(&self, path: &str) -> Response {
        let mut stream = TcpStream::connect(self.addr).expect("connecting to tileserver");
        stream
            .set_read_timeout(Some(IO_TIMEOUT))
            .expect("read timeout");
        stream
            .set_write_timeout(Some(IO_TIMEOUT))
            .expect("write timeout");
        write!(
            stream,
            "GET {path} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
            self.addr
        )
        .expect("sending request");
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).expect("reading response");
        Response::parse(&raw)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// An address on loopback no one listens on right now.
fn free_local_addr() -> SocketAddr {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .expect("reserving a free port")
}

struct Response {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Response {
    /// Parse a `Connection: close` HTTP/1.1 response (no chunked bodies:
    /// axum sends fixed-length bodies for these routes).
    fn parse(raw: &[u8]) -> Self {
        let split = raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .unwrap_or_else(|| panic!("no header terminator in {raw:?}"));
        let head = std::str::from_utf8(&raw[..split]).expect("UTF-8 response head");
        let mut lines = head.split("\r\n");
        let status_line = lines.next().expect("status line");
        let status = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("bad status line {status_line:?}"));
        let headers = lines
            .filter_map(|l| l.split_once(':'))
            .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_owned()))
            .collect();
        Self {
            status,
            headers,
            body: raw[split + 4..].to_vec(),
        }
    }

    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

/// The raster source `style.json` advertises.
fn raster_source(server: &Server) -> serde_json::Value {
    let resp = server.get("/style.json");
    assert_eq!(resp.status, 200, "GET /style.json");
    let style: serde_json::Value = serde_json::from_slice(&resp.body).expect("style.json is JSON");
    style["sources"]["raster"].clone()
}

/// Path prefix of the style's raster tile URL template
/// `<base>/tiles/<build id>/{z}/{x}/{y}`: `/tiles/<build id>`.
fn tile_path_prefix(server: &Server) -> String {
    let source = raster_source(server);
    let url = source["tiles"][0]
        .as_str()
        .unwrap_or_else(|| panic!("no raster tile URL in source: {source}"));
    let start = url
        .find("/tiles/")
        .unwrap_or_else(|| panic!("unexpected tile URL {url:?}"));
    // Drop the trailing `/{z}/{x}/{y}` placeholders.
    let path = &url[start..];
    let prefix = path
        .rsplitn(4, '/')
        .nth(3)
        .unwrap_or_else(|| panic!("tile URL {url:?} lacks z/x/y segments"));
    prefix.to_owned()
}

fn tile_path(prefix: &str, z: u8, (x, y): (u32, u32)) -> String {
    format!("{prefix}/{z}/{x}/{y}")
}

#[test]
fn tile_coordinates_are_validated_before_rendering() {
    let dir = Scratch::new("tileserver-tile-coords");
    let [coarse, fine] = common::write_raster_fixtures(&dir);
    let server = Server::spawn(&[
        "--cache-mb".as_ref(),
        "0".as_ref(),
        coarse.as_os_str(),
        fine.as_os_str(),
    ]);
    let prefix = tile_path_prefix(&server);
    let source = raster_source(&server);
    let zoom = |key: &str| -> u8 {
        source[key]
            .as_u64()
            .and_then(|z| u8::try_from(z).ok())
            .unwrap_or_else(|| panic!("no {key} in raster source: {source}"))
    };
    let (min_zoom, max_zoom) = (zoom("minzoom"), zoom("maxzoom"));
    let inside = |z| tile_containing(10.3, 56.3, z);

    // Beyond any Web Mercator zoom a tile can exist at.
    assert_eq!(server.get(&tile_path(&prefix, 31, (0, 0))).status, 400);
    assert_eq!(server.get(&tile_path(&prefix, 255, (0, 0))).status, 400);

    // Column/row off the edge of the zoom's 2^z × 2^z grid.
    let z = max_zoom;
    let edge = 1u32 << z;
    assert_eq!(server.get(&tile_path(&prefix, z, (edge, 0))).status, 400);
    assert_eq!(server.get(&tile_path(&prefix, z, (0, edge))).status, 400);
    assert_eq!(
        server
            .get(&tile_path(&prefix, z, (edge - 1, edge - 1)))
            .status,
        204,
        "last tile of the grid is valid (and far from every chart)"
    );

    // Valid tiles outside the zoom range `style.json` advertises.
    let resp = server.get(&tile_path(&prefix, max_zoom + 1, inside(max_zoom + 1)));
    assert_eq!(resp.status, 404, "zoom above the advertised maxzoom");
    assert_eq!(
        resp.header("cache-control"),
        None,
        "an out-of-range miss is not an immutable tile answer"
    );
    if let Some(below) = min_zoom.checked_sub(1) {
        let resp = server.get(&tile_path(&prefix, below, inside(below)));
        assert_eq!(resp.status, 404, "zoom below the advertised minzoom");
    }

    // The server survived all of the above and still renders.
    assert_eq!(
        server
            .get(&tile_path(&prefix, max_zoom, inside(max_zoom)))
            .status,
        200
    );
}

#[test]
fn sparse_raster_server_answers_data_omitted_and_empty_tiles() {
    let dir = Scratch::new("tileserver-sparse-raster");
    let [coarse, fine] = common::write_raster_fixtures(&dir);
    let server = Server::spawn(&[
        "--sparse".as_ref(),
        "--cache-mb".as_ref(),
        "0".as_ref(),
        coarse.as_os_str(),
        fine.as_os_str(),
    ]);
    let prefix = tile_path_prefix(&server);

    // Coarse cell at its own native (= detail) zoom: real PNG data.
    let centre = (
        f64::midpoint(COARSE_BOX[0], COARSE_BOX[2]),
        f64::midpoint(COARSE_BOX[1], COARSE_BOX[3]),
    );
    let resp = server.get(&tile_path(
        &prefix,
        COARSE_ZOOM,
        tile_containing(centre.0, centre.1, COARSE_ZOOM),
    ));
    assert_eq!(resp.status, 200, "coarse tile at native zoom");
    assert_eq!(resp.header("content-type"), Some("image/png"));
    assert!(
        resp.body.starts_with(b"\x89PNG\r\n\x1a\n"),
        "body is not a PNG"
    );

    // Coarse-only area past the coarse cell's detail zoom: omitted.
    let coarse_only = (10.8, 56.1);
    assert!(
        coarse_only.0 > FINE_BOX[2] + 0.04,
        "fixture: point must be clear of the fine cell's z10 tile"
    );
    let resp = server.get(&tile_path(
        &prefix,
        10,
        tile_containing(coarse_only.0, coarse_only.1, 10),
    ));
    assert_eq!(resp.status, 404, "coarse-only tile past its detail zoom");
    assert!(
        resp.header("cache-control").is_some(),
        "a sparse omission is a cacheable tile answer, not a routing miss"
    );

    // Nothing anywhere near: empty.
    let resp = server.get(&tile_path(&prefix, 10, tile_containing(50.0, 0.0, 10)));
    assert_eq!(resp.status, 204, "tile far away from every chart");
}

/// `GET /style.json` over cleartext HTTP/2 with prior knowledge (h2c), the
/// way an h2c reverse proxy talks to its upstream. HTTP/2 carries the host
/// in the `:authority` pseudo-header, not a `Host` header. Hand-rolled
/// frames, uncompressed HPACK literals: just enough HTTP/2 to read one
/// response body.
fn h2c_get_body(addr: SocketAddr, path: &str) -> Vec<u8> {
    const DATA: u8 = 0x0;
    const HEADERS: u8 = 0x1;
    const RST_STREAM: u8 = 0x3;
    const SETTINGS: u8 = 0x4;
    const GOAWAY: u8 = 0x7;
    const END_STREAM: u8 = 0x1;
    const END_HEADERS: u8 = 0x4;
    const ACK: u8 = 0x1;

    fn frame(kind: u8, flags: u8, stream: u32, payload: &[u8]) -> Vec<u8> {
        let len = u32::try_from(payload.len()).unwrap().to_be_bytes();
        let mut out = vec![len[1], len[2], len[3], kind, flags];
        out.extend_from_slice(&stream.to_be_bytes());
        out.extend_from_slice(payload);
        out
    }
    /// HPACK "literal header field without indexing", indexed name.
    fn literal(name_index: u8, value: &str) -> Vec<u8> {
        let len = u8::try_from(value.len()).unwrap();
        assert!(len < 0x7f, "single-byte HPACK length only");
        let mut out = vec![name_index, len];
        out.extend_from_slice(value.as_bytes());
        out
    }

    let authority = addr.to_string();
    // Static table: 2 = `:method: GET`, 6 = `:scheme: http`,
    // name 4 = `:path`, name 1 = `:authority`.
    let mut block = vec![0x82, 0x86];
    block.extend(literal(0x04, path));
    block.extend(literal(0x01, &authority));

    let mut stream = TcpStream::connect(addr).expect("connecting to tileserver");
    stream
        .set_read_timeout(Some(IO_TIMEOUT))
        .expect("read timeout");
    let mut request = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n".to_vec();
    request.extend(frame(SETTINGS, 0, 0, &[]));
    request.extend(frame(HEADERS, END_STREAM | END_HEADERS, 1, &block));
    stream.write_all(&request).expect("sending h2c request");

    let mut body = Vec::new();
    loop {
        let mut head = [0u8; 9];
        stream.read_exact(&mut head).expect("reading frame header");
        let len = usize::from(head[0]) << 16 | usize::from(head[1]) << 8 | usize::from(head[2]);
        let (kind, flags) = (head[3], head[4]);
        let id = u32::from_be_bytes([head[5], head[6], head[7], head[8]]) & 0x7fff_ffff;
        let mut payload = vec![0u8; len];
        stream
            .read_exact(&mut payload)
            .expect("reading frame payload");
        match kind {
            SETTINGS if flags & ACK == 0 => stream
                .write_all(&frame(SETTINGS, ACK, 0, &[]))
                .expect("acking settings"),
            RST_STREAM | GOAWAY => panic!("server aborted h2c request: frame type {kind}"),
            DATA | HEADERS if id == 1 => {
                if kind == DATA {
                    body.extend_from_slice(&payload);
                }
                if flags & END_STREAM != 0 {
                    return body;
                }
            }
            _ => {}
        }
    }
}

#[test]
fn style_tile_url_uses_the_requested_authority_over_http2() {
    let dir = Scratch::new("tileserver-h2c-style");
    let [coarse, fine] = common::write_raster_fixtures(&dir);
    let server = Server::spawn(&[
        "--cache-mb".as_ref(),
        "0".as_ref(),
        coarse.as_os_str(),
        fine.as_os_str(),
    ]);

    let body = h2c_get_body(server.addr, "/style.json");
    let style: serde_json::Value = serde_json::from_slice(&body)
        .unwrap_or_else(|e| panic!("style.json is JSON ({e}): {body:?}"));
    let url = style["sources"]["raster"]["tiles"][0]
        .as_str()
        .unwrap_or_else(|| panic!("no raster tile URL in style: {style}"));
    let expected = format!("http://{}/tiles/", server.addr);
    assert!(
        url.starts_with(&expected),
        "h2 style.json must point tiles at the requested authority {expected:?}, got {url:?}"
    );
}
