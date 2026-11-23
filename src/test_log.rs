//! Test-only capture of `tracing` output, for asserting that a code path
//! logs (or stays silent).

use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Capture {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("capture lock").extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Run `f` under a thread-local subscriber; return its result and the
/// formatted (ANSI-free) log lines it emitted.
pub fn capture_logs<T>(f: impl FnOnce() -> T) -> (T, String) {
    let capture = Capture::default();
    let writer = capture.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(move || writer.clone())
        .with_ansi(false)
        .finish();
    let out = tracing::subscriber::with_default(subscriber, f);
    let bytes = capture.0.lock().expect("capture lock").clone();
    (out, String::from_utf8(bytes).expect("log is UTF-8"))
}
