# Modernization: drop `probabilistic-collections` from the MLT encoder chain

Fingerprint: `quilt-tiler:mlt-core-transitive-deps:unmaintained-library`
Evidence re-verified 2026-09-29; chain re-confirmed on master 2026-10-10 (`Cargo.lock` still
pins mlt-core 0.12.9 → probabilistic-collections 0.7.0 → bincode 1.3.3). Revisit by 2026-12-31.

## Current state

`mlt-core` (our MLT encoder, `src/s57_source.rs` → `TileLayer::encode`) has a
normal, non-optional dependency on `probabilistic-collections = "^0.7"`. That crate is
abandoned and pulls in `bincode 1.3.3`, which RustSec flags as unmaintained.

```
bincode v1.3.3
└── probabilistic-collections v0.7.0
    └── mlt-core v0.12.9
        └── quilt-tiler v0.1.0
```

| Claim | Evidence (2026-09-29) |
|---|---|
| `probabilistic-collections` is abandoned | crates.io: max version 0.7.0, published 2020-05-11, no later release. GitLab `jeffrey-xiao/probabilistic-collections-rs`: last commit 2023-01-06 ("Merge branch 'clippy-25-nov-2022'"). |
| `bincode` is flagged | [RUSTSEC-2025-0141](https://rustsec.org/advisories/RUSTSEC-2025-0141.html): "Bincode is unmaintained", type INFO Unmaintained, reported 2025-12-16, issued 2026-01-07, no patched versions. The bincode team calls 1.3.3 "a complete version … not in need of any updates". |
| Upgrading `mlt-core` does not help | `mlt-core` 0.12.9, 0.13.0, 0.14.5, 0.15.1 and 0.16.0 (latest, 2026-09-28) all declare `probabilistic-collections ^0.7`, `kind = normal`, `optional = false`. `rust/mlt-core/Cargo.toml` on upstream `main` still has `probabilistic-collections.workspace = true`. |
| Nobody has raised it upstream | GitHub search of `maplibre/maplibre-tile-spec` for "bincode": 0 results; for "probabilistic": 5 unrelated PRs. |

### Actual exposure (narrower than the advisory suggests)

- **bincode is dead code for us.** Every `bincode` reference in `probabilistic-collections`
  0.7.0 sits inside a `#[cfg(test)]` module (serde round-trip tests), yet it is declared as a
  normal dependency. It is compiled into our build and never called.
- **rand 0.7 / getrandom 0.1 are also unused on our path.** `probabilistic-collections` uses
  `rand` for `SipHasherBuilder::from_entropy()` and the cuckoo / dd-bloom filters. `mlt-core`
  uses none of those; it seeds everything with `SipHasherBuilder::from_seed`.
- **What `mlt-core` actually executes** (both files under `src/encoder/`):
  - `geometry/streams.rs::dict_may_be_beneficial`:
    `HyperLogLog::<Coord<i32>>::with_hasher(0.03, SipHasherBuilder::from_seed(0, 0))`,
    then `insert` and `len()`. This is a gate that decides whether to try Hilbert/Morton
    vertex dictionaries.
  - `property/shared_dict.rs::group_string_properties`:
    `MinHash::with_hashers(128, [from_seed(0,0), from_seed(1,1)])`, then `get_min_hashes`
    over exact values and byte trigrams. This decides which string columns share a dictionary.
  - None of these types appear in `mlt-core`'s public API (`dict_may_be_beneficial` is
    `pub(super)`, `group_string_properties` is `pub(crate)`).
- Dropping `probabilistic-collections` removes 9 crates from our graph:
  `probabilistic-collections`, `bincode`, `rand 0.7.3`, `rand_chacha 0.2.2`, `rand_core 0.5.1`,
  `rand_xorshift 0.2.0`, `getrandom 0.1.16`, `ppv-lite86`, `siphasher 0.3.11`
  (per `cargo tree -i <crate> --depth 1`; `byteorder` stays because `hash32` uses it).

No known vulnerability exists in any of these today. The risk is that nothing gets fixed if one
turns up, and the RustSec advisory makes any audit tooling fail.

### Latent upstream bug found while sizing the port

`probabilistic-collections` 0.7.0's `HyperLogLog` is broken in two ways:

- `get_estimate()` computes `1 / (alpha * m² * Σ 2^-M)` instead of `alpha * m² / Σ 2^-M`. The
  raw estimate is therefore always below `2.5 * m`, so `len()` always takes the linear-counting
  branch `m * ln(m / zeros)`. That branch returns `inf` once every register is non-zero.
- `with_hasher` sizes the register array with `ln` where it should use `log2`
  (`p = ceil(ln((1.04/0.03)²)) = 8`, so `m = 256` instead of `2^11`).

As a result, in `mlt-core`'s `dict_may_be_beneficial`, `hll.len()` is `inf` for any geometry
column with more than roughly 1.5k unique vertices. The clamp turns that into
`coord_count`, the uniqueness ratio becomes 1.0, and the Hilbert/Morton vertex-dictionary
layouts are never tried, however repetitive the vertices are. A throwaway probe
(`probabilistic-collections = "=0.7.0"`, the same call as `mlt-core` 0.16.0, each unique
coordinate inserted 3 times, so the true ratio is 0.33) printed:

```
  coords   unique      hll.len()    ratio      dict?
     300      100          107.8    0.359       true
    3000     1000         1064.7    0.355       true
    4500     1500            inf    1.000      false
   60000    20000            inf    1.000      false
```

Searching upstream for "HyperLogLog" and "dict_may_be_beneficial" turned up no issue about this.
For quilt-tiler, the likely effect is that larger layers in each tile miss vertex-dictionary
compression `[INFERENCE: not measured on real charts]`. Output is still valid MLT.

## Local gate (already on master)

- `deny.toml`, `cargo deny check` in `just check` and in `.github/workflows/ci.yml` (job
  "Dependencies — deny + machete") on every push to `master` and every PR, over the
  all-features graph.
- `RUSTSEC-2025-0141` is ignored there explicitly ("transitive via mlt-core; no upstream fix").
- `anyhow` (RUSTSEC-2026-0190) and `h2` (RUSTSEC-2026-0258) are already bumped past their
  advisories on master.

## Target state

`mlt-core` no longer depends on `probabilistic-collections`, quilt-tiler is on that
`mlt-core` release, `cargo tree -i bincode` is empty, and the `deny.toml` ignore is gone.

## Migration steps

1. **File the upstream issue** on `maplibre/maplibre-tile-spec` (draft below) and put its URL into
   the `reason` of the `deny.toml` ignore.
2. **Upstream PR A: drop the dependency without changing behavior.** Port the two estimators
   into `mlt-core`, around 150 lines plus tests, on top of `siphasher` 1.x (1.0.4, released
   2026-09-25, actively maintained):
   - `SipHasherBuilder::from_seed(k0, k1)` becomes a `BuildHasher` wrapping
     `siphasher::sip::SipHasher::new_with_keys(k0, k1)`. The algorithm is still SipHash-2-4,
     so hash values stay the same.
   - MinHash: hash each item with both seeded hashers into `(a, b)`, then for
     `i in 0..128` take `min` over items of the enhanced double-hash sequence
     (`a`, then `a += b; b += c; c += 1`, all wrapping). That is the whole of
     `probabilistic-collections`' `DoubleHasher`/`HashIter` + `get_min_hashes`.
   - HyperLogLog: reproduce 0.7.0 exactly, including both bugs above, so that
     `dict_may_be_beneficial` decides exactly as it does today. Mark the bugs with a comment
     that points to PR B.
   - **Keep it bit-exact.** Before deleting the old dependency, add a test that runs old and new
     implementations on the same fixed inputs and asserts identical `Vec<u64>` MinHash signatures
     and identical `len()` values. Then remove that test together with the dependency, or keep
     the recorded vectors as golden values. Tile bytes stay unchanged.
   - **Feasibility is proven.** A throwaway prototype of exactly this port (appendix) on
     `siphasher` 1.0.4 was compared against `probabilistic-collections` 0.7.0, which uses
     `siphasher` 0.3.11. It produced bit-identical HLL `len()` for 8 inputs (100–20,000 unique
     `Coord<i32>`, including the `inf` cases) and identical 128×`u64` MinHash signatures for
     10 cases (exact `&str` values and `[u8; 3]` trigrams).
   - Alternatives if upstream prefers a dependency: `cardinality-estimator` 1.0.3 (updated
     2026-02-11) for HLL and `probminhash` 0.1.12 (updated 2025-06-11) for MinHash. These are
     not bit-exact, so this becomes PR A and PR B at once.
3. **Upstream PR B: fix the HLL gate (behavior change; upstream decides).** Either use a correct
   HLL estimate or, if benchmarks allow, count exactly. The gate exists because racing the
   dictionary layouts unconditionally was "~2× slower overall" (code comment on
   `dict_may_be_beneficial`). This changes which vertex layouts get raced and picked for columns
   with more than about 1.5k unique vertices: tile bytes change, probably get smaller, and
   encoding gets slower. It needs upstream benchmarks. It does not block steps 4–5.
4. **Bump `mlt-core` here** to the first release without the dependency. Evidence that this is
   cheap: on the 2026-09-29 base, bumping 0.12 → 0.16.0 compiled without source changes,
   `cargo test --workspace` passed (162 tests), and `cargo clippy --workspace --all-targets`
   reported the same warning set as on 0.12.9 (checked in a scratch copy, not committed).
5. **Remove the `RUSTSEC-2025-0141` entry from `deny.toml`.** Confirm `cargo deny check` passes
   and `cargo tree -i bincode` reports nothing.

If upstream declines or stalls past the revisit date: keep the ignore, since bincode is
test-only in this chain. Patching in a fork of `probabilistic-collections` with `bincode` moved to
`[dev-dependencies]` would work (`[patch.crates-io]`), but then we maintain a fork of an
abandoned crate for an INFO-level advisory. Not recommended. Vendoring `mlt-core` is not
worth it either.

## What breaks, and for whom

- **quilt-tiler, after PR A:** no code change. Step 4 is a `Cargo.toml` requirement change plus
  lockfile, and tile bytes stay identical.
- **quilt-tiler, after PR B (or a non-bit-exact PR A):** `.pmtiles` output bytes change for
  layers with many unique vertices, and tiling time may go up. Our tests only compare encoder
  output against itself (determinism and layer ordering), not golden bytes, so they keep passing.
  Consumers get a different but valid tile. Profile a full tiling run before and after
  adopting that release (per repo policy, performance claims need profiling data).
- **mlt-core users:** nothing public changes. The types are internal, so this is a non-breaking
  upstream change.

## Effort estimate

- Upstream PR A plus equivalence tests: about half a day to a day, including review turnaround.
  Upstream is very active (11 releases, 0.12.9 → 0.16.0, between 2026-09-07 and 2026-09-28).
- Upstream PR B: mostly benchmarking, and upstream's call.
- Local follow-up (steps 4–5): under an hour, going by the 0.16 bump check above. Add a
  before/after profiling run if the release also contains PR B.

## Recommendation

File the issue, which covers both the dependency and the HLL bug, and offer PR A. PR A
resolves the advisory with zero behavior change and is easy to review. PR B needs upstream's
benchmarks and judgment. Keep the documented ignore until a release without the dependency lands.

## Draft upstream issue (maplibre/maplibre-tile-spec)

> **rust/mlt-core: replace unmaintained `probabilistic-collections` (pulls in `bincode` 1.3.3, RUSTSEC-2025-0141)**
>
> `mlt-core` depends on `probabilistic-collections = "^0.7"` (normal, non-optional, still on
> `main` and in 0.16.0). Its last release was 0.7.0 on 2020-05-11, and the GitLab repo's last
> commit was 2023-01-06. It declares `bincode ^1.0` as a normal dependency, although bincode is
> only used in its `#[cfg(test)]` modules. As a result every `mlt-core` user gets
> RUSTSEC-2025-0141 ("bincode is unmaintained", no patched version) from `cargo audit` /
> `cargo deny`, plus an otherwise unused 2020-era `rand 0.7` / `rand_core 0.5` /
> `getrandom 0.1` stack.
>
> `mlt-core` uses only:
> - `HyperLogLog::with_hasher(0.03, SipHasherBuilder::from_seed(0, 0))` + `insert` + `len` in
>   `encoder/geometry/streams.rs::dict_may_be_beneficial`
> - `MinHash::with_hashers(128, [from_seed(0,0), from_seed(1,1)])` + `get_min_hashes` in
>   `encoder/property/shared_dict.rs::group_string_properties`
>
> Proposal: port these two estimators into `mlt-core` on top of `siphasher` 1.x
> (about 150 lines). SipHash-2-4 with the same keys and the same enhanced double hashing gives
> bit-identical signatures and estimates, so encoder output does not change. A temporary
> equivalence test against the old crate can prove that before the dependency is removed. This
> drops 9 crates from downstream dependency graphs. Happy to send the PR.
>
> Separately, while sizing this: the 0.7.0 `HyperLogLog` is buggy. `get_estimate()` computes
> `1 / (alpha * m² * Σ 2^-M)` instead of `alpha * m² / Σ 2^-M`, and `with_hasher` uses `ln`
> where it should use `log2` for the register count (`m = 256` at 0.03). `len()` therefore always
> falls back to linear counting and returns `inf` once all 256 registers are set. In practice
> that means more than about 1.5k unique vertices. `dict_may_be_beneficial` then clamps to
> ratio 1.0, so the Hilbert/Morton dictionary layouts are never tried for such columns, even
> when every vertex repeats 3× (probe: 1,000 unique → `len()` 1064.7, ratio 0.355; 1,500
> unique → `inf`, ratio 1.0). The port above could keep this behavior bit-for-bit and leave
> the fix to a follow-up with benchmarks, or fix it in the same change. Your call.

## Appendix: verified prototype of PR A's core

This throwaway code (`siphasher = "1"`) was checked bit-for-bit against
`probabilistic-collections` 0.7.0 as described in step 2. The `unwrap()` mirrors the old crate,
which panics on empty input; both `mlt-core` call sites already guard against that.

```rust
use std::hash::{BuildHasher, Hash, Hasher};
use siphasher::sip::SipHasher;

/// Replaces `SipHasherBuilder::from_seed(k0, k1)`.
#[derive(Clone, Copy)]
struct Sip(u64, u64);
impl BuildHasher for Sip {
    type Hasher = SipHasher;
    fn build_hasher(&self) -> SipHasher { SipHasher::new_with_keys(self.0, self.1) }
}

/// Replaces `MinHash::with_hashers(n, [s0, s1]).get_min_hashes(items)`.
fn min_hashes<T: Hash>(n: usize, s: [Sip; 2], items: impl Iterator<Item = T>) -> Vec<u64> {
    let mut iters: Vec<(u64, u64, u64)> = items
        .map(|it| {
            let (mut h1, mut h2) = (s[0].build_hasher(), s[1].build_hasher());
            it.hash(&mut h1);
            it.hash(&mut h2);
            (h1.finish(), h2.finish(), 0)
        })
        .collect();
    (0..n)
        .map(|_| {
            iters
                .iter_mut()
                .map(|(a, b, c)| {
                    let ret = *a;
                    *a = a.wrapping_add(*b);
                    *b = b.wrapping_add(*c);
                    *c += 1;
                    ret
                })
                .min()
                .unwrap()
        })
        .collect()
}

/// Replaces `HyperLogLog::with_hasher(eps, s)` + `insert` + `len`, 0.7.0 bugs included.
fn hll_len<T: Hash>(eps: f64, s: Sip, items: impl Iterator<Item = T>) -> f64 {
    let p = (1.04 / eps).powi(2).ln().ceil() as usize; // BUG kept: ln, not log2
    let alpha = match p {
        4 => 0.673,
        5 => 0.697,
        6 => 0.709,
        p => 0.7213 / (1.0 + 1.079 / f64::from(1u32 << p)),
    };
    let mut regs = vec![0u8; 1 << p];
    for it in items {
        let mut h = s.build_hasher();
        it.hash(&mut h);
        let hash = h.finish();
        let idx = hash as usize & (regs.len() - 1);
        let v = (!hash >> p).trailing_zeros() as u8;
        regs[idx] = regs[idx].max(v + 1);
    }
    let m = regs.len() as f64;
    // BUG kept: inverted estimate, so the linear-counting branch below always runs.
    let est = 1.0 / (alpha * m * m * regs.iter().map(|v| 1.0 / 2.0f64.powi(i32::from(*v))).sum::<f64>());
    if est <= 2.5 * m {
        let zeros = regs.iter().filter(|v| **v == 0).count() as u64;
        m * (m / zeros as f64).ln()
    } else if est <= 1.0 / 3.0 * 2.0f64.powi(32) {
        est
    } else {
        -(2.0f64.powi(32)) * (1.0 - est / 2.0f64.powi(32)).ln()
    }
}
```
