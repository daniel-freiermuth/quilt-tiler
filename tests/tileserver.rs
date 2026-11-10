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

/// Path prefix of the style's raster tile URL template
/// `<base>/tiles/<build id>/{z}/{x}/{y}`: `/tiles/<build id>`.
fn tile_path_prefix(server: &Server) -> String {
    let resp = server.get("/style.json");
    assert_eq!(resp.status, 200, "GET /style.json");
    let style: serde_json::Value = serde_json::from_slice(&resp.body).expect("style.json is JSON");
    let url = style["sources"]["raster"]["tiles"][0]
        .as_str()
        .unwrap_or_else(|| panic!("no raster tile URL in style: {style}"));
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
