//! Size checks on Matroska/WebM elements before symphonia reads them: it allocates whatever an
//! element, a laced frame or a Vorbis codec header claims, and a failed allocation aborts the
//! process.

use crate::{AudioError, Result};

/// Elements symphonia descends into rather than reads whole.
const MASTERS: [u32; 29] = [
    0x1A45_DFA3, // EBML
    0x1853_8067, // Segment
    0x114D_9B74, // SeekHead
    0x4DBB,      // Seek
    0x1549_A966, // Info
    0x1F43_B675, // Cluster
    0xA0,        // BlockGroup
    0x75A1,      // BlockAdditions
    0xA6,        // BlockMore
    0x1654_AE6B, // Tracks
    TRACK_ENTRY,
    0xE0,        // Video
    0xE1,        // Audio
    0x6D80,      // ContentEncodings
    0x6240,      // ContentEncoding
    0x5035,      // ContentEncryption
    0x47E7,      // ContentEncAesSettings
    0x55B0,      // Colour
    0x55D0,      // MasteringMetadata
    0x1C53_BB6B, // Cues
    0xBB,        // CuePoint
    0xB7,        // CueTrackPositions
    0x1043_A770, // Chapters
    0x45B9,      // EditionEntry
    0xB6,        // ChapterAtom
    0x80,        // ChapterDisplay
    0x1254_C367, // Tags
    0x7373,      // Tag
    0x67C8,      // SimpleTag
];

/// Where symphonia picks up again after an element ID it can't read: Cluster, Cues, Info,
/// SeekHead, Tags, Tracks.
const RESYNC: [u32; 6] = [0x1F43_B675, 0x1C53_BB6B, 0x1549_A966, 0x114D_9B74, 0x1254_C367, 0x1654_AE6B];

const TRACK_ENTRY: u32 = 0xAE;
const SIMPLE_BLOCK: u32 = 0xA3;
const BLOCK: u32 = 0xA1;
const CODEC_ID: u32 = 0x86;
const CODEC_PRIVATE: u32 = 0x63A2;

/// Deeper than any real file nests (SimpleTag may nest in itself).
const MAX_DEPTH: usize = 32;

/// Refuse the Matroska stream at `at` if an element, a laced frame or a Vorbis header claims more
/// bytes than the file holds.
pub(crate) fn check(b: &[u8], mut at: usize) -> Result<()> {
    // Each resync point lies past the one before, so this ends.
    while let Some(next) = elements(b, at, b.len(), 0)? {
        at = next;
    }
    Ok(())
}

fn malformed(what: &str) -> AudioError {
    AudioError::Malformed(format!("Matroska {what} is larger than the file"))
}

/// An EBML variable-size integer at `pos` and its width, read as symphonia reads it: the length
/// marker cleared, a lone 0xFF as `u64::MAX` (its "unknown size"), and a zero first byte (no
/// marker) as `u64::MAX` too, since symphonia makes that a size near 2^63. None if the bytes run
/// out first.
fn vint(b: &[u8], pos: usize) -> Option<(u64, usize)> {
    let first = *b.get(pos)?;
    if first == 0xFF {
        return Some((u64::MAX, 1));
    }
    let width = first.leading_zeros() as usize;
    if width > 7 {
        return b.get(pos + 8).map(|_| (u64::MAX, 9));
    }
    let mut v = u64::from(first) & (0xFF >> (width + 1));
    for i in 1..=width {
        v = (v << 8) | u64::from(*b.get(pos + i)?);
    }
    Some((v, width + 1))
}

/// Element ID at `pos` (1 to 4 bytes, marker kept), or None where symphonia would resync.
fn id(b: &[u8], pos: usize) -> Option<(u32, usize)> {
    let first = *b.get(pos)?;
    let width = first.leading_zeros() as usize;
    if width > 3 {
        return None;
    }
    let mut v = u32::from(first);
    for i in 1..=width {
        v = (v << 8) | u32::from(*b.get(pos + i)?);
    }
    Some((v, width + 1))
}

/// Walk the elements in `pos..end`, as symphonia reads them. `Some(at)`: an ID symphonia can't read
/// sends it scanning on to the next top-level element, at `at`, to read from there.
fn elements(b: &[u8], mut pos: usize, end: usize, depth: usize) -> Result<Option<usize>> {
    if depth > MAX_DEPTH {
        return Err(AudioError::Malformed("Matroska elements are nested too deeply".into()));
    }
    // A TrackEntry's codec, to check its private data once both are known.
    let (mut codec, mut private) = (None, None);
    while pos < end {
        let Some((tag, tag_len)) = id(b, pos) else { return Ok(resync(b, pos + 1)) };
        // A file cut short inside a header: symphonia stops there too, before allocating.
        let Some((size, size_len)) = vint(b, pos + tag_len) else { return Ok(None) };
        let unknown = size == u64::MAX && size_len == 1;
        let body = pos + tag_len + size_len;
        if MASTERS.contains(&tag) {
            // Its children are read one by one, up to its end or (unknown or huge size) its parent's.
            let stop = usize::try_from(size).ok().and_then(|s| body.checked_add(s)).map_or(end, |e| e.min(end));
            if let Some(at) = elements(b, body, stop, depth + 1)? {
                return Ok(Some(at));
            }
            pos = stop;
            continue;
        }
        // Symphonia reads an unknown size as 0.
        let size = if unknown { 0 } else { size };
        let rest = b.len().saturating_sub(body);
        if size > rest as u64 {
            return Err(malformed("element"));
        }
        // `size <= rest`, so it fits a usize.
        let data = b.get(body..body + size as usize).unwrap_or_default();
        match tag {
            SIMPLE_BLOCK | BLOCK => laced_frames(data)?,
            CODEC_ID => codec = Some(data),
            CODEC_PRIVATE => private = Some(data),
            _ => {}
        }
        pos = body + data.len();
    }
    if let (Some(c), Some(p)) = (codec, private)
        && c.split(|&x| x == 0).next() == Some(b"A_VORBIS".as_slice())
    {
        vorbis_private(p)?;
    }
    Ok(None)
}

/// Where symphonia picks up after an unreadable ID: the next top-level element ID from `from` on.
fn resync(b: &[u8], from: usize) -> Option<usize> {
    let ids = b.get(from..)?.windows(4);
    ids.map(|w| w.iter().fold(0u32, |v, &x| (v << 8) | u32::from(x))).position(|id| RESYNC.contains(&id)).map(|i| from + i)
}

/// Xiph-laced sizes (runs of 255 plus a final byte), `count` of them, starting at `pos`.
fn xiph_sizes(d: &[u8], mut pos: usize, count: usize) -> Result<(Vec<u64>, usize)> {
    let mut sizes = Vec::new();
    let mut run = 0u64;
    while sizes.len() < count {
        let byte = u64::from(*d.get(pos).ok_or_else(|| malformed("laced frame"))?);
        pos += 1;
        run += byte;
        if byte != 255 {
            sizes.push(run);
            run = 0;
        }
    }
    Ok((sizes, pos))
}

/// Each size must fit in what is left of `d` after the ones before it: symphonia allocates a
/// frame before reading it.
fn fit(d: &[u8], pos: usize, sizes: &[u64]) -> Result<()> {
    let mut rest = d.len().saturating_sub(pos) as u64;
    for &s in sizes {
        rest = rest.checked_sub(s).ok_or_else(|| malformed("laced frame"))?;
    }
    Ok(())
}

/// A block's laced frame sizes (Xiph or EBML lacing), against the block's own bytes.
fn laced_frames(d: &[u8]) -> Result<()> {
    // Track number, 16-bit timestamp, flags.
    let Some((_, track_len)) = vint(d, 0) else { return Ok(()) };
    let Some(&flags) = d.get(track_len + 2) else { return Ok(()) };
    let mut pos = track_len + 3;
    let lacing = (flags >> 1) & 0b11;
    if lacing == 0b00 || lacing == 0b10 {
        // None, or fixed sizes splitting the block evenly.
        return Ok(());
    }
    let Some(&frames) = d.get(pos) else { return Ok(()) };
    pos += 1;
    let sizes = if lacing == 0b01 {
        let (sizes, after) = xiph_sizes(d, pos, usize::from(frames))?;
        pos = after;
        sizes
    } else {
        let mut sizes: Vec<u64> = Vec::new();
        for _ in 0..frames {
            // Running out of bytes here is an error symphonia reports before allocating.
            let Some((v, len)) = vint(d, pos) else { return Ok(()) };
            pos += len;
            // Later sizes are signed differences from the one before (symphonia wraps).
            let size = match sizes.last() {
                None => v,
                Some(&last) => {
                    let half = (1i64 << (len * 7 - 1)) - 1;
                    (last as i64).wrapping_add((v as i64).wrapping_sub(half)) as u64
                }
            };
            sizes.push(size);
        }
        sizes
    };
    fit(d, pos, &sizes)
}

/// Vorbis codec private data: a packet count and Xiph-laced packet sizes.
fn vorbis_private(d: &[u8]) -> Result<()> {
    let Some(&count) = d.first() else { return Ok(()) };
    let (sizes, pos) = xiph_sizes(d, 1, usize::from(count))?;
    fit(d, pos, &sizes)
}
