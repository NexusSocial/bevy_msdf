use bevy::math::{uvec2, UVec2, Vec2};
use msdfgen::{Bitmap, FillRule, FontExt, MsdfGeneratorConfig, Range, Rgba, Shape};
use owned_ttf_parser::{Face, GlyphId};
use rect_packer::Packer;

use crate::MsdfAtlasLoaderError;

pub type FontResult<T> = Result<T, MsdfAtlasLoaderError>;

#[derive(Copy, Clone, Debug)]
pub struct GlyphVertex {
    pub position: Vec2,
    pub tex_coords: Vec2,
}

pub struct GlyphInfo {
    pub position: UVec2,
    pub size: UVec2,
    pub anchor: Vec2,
    pub shape: GlyphShape,
    pub vertices: [GlyphVertex; 4],
}

pub struct GlyphAtlas {
    pub width: u32,
    pub height: u32,
    pub glyphs: Vec<Option<GlyphInfo>>,
}

impl GlyphAtlas {
    pub const PX_PER_EM: f64 = 24.0;
    pub const RANGE: Range<f64> = Range::Px(8.0);
    pub const ANGLE_THRESHOLD: f64 = 3.0;

    /// Creates a [GlyphAtlas] for a given [Face].
    ///
    /// Returns the completed atlas and a list of all glyph IDs that non-fatally
    /// failed to load.
    pub fn new(face: &Face) -> FontResult<(GlyphAtlas, Vec<GlyphId>)> {
        let mut glyphs = Vec::with_capacity(face.number_of_glyphs() as usize);
        let mut glyph_shape_errors = vec![];
        for c in 0..face.number_of_glyphs() {
            let glyph = GlyphShape::new(
                face.units_per_em() as f64,
                Self::PX_PER_EM,
                Self::RANGE,
                Self::ANGLE_THRESHOLD,
                face,
                GlyphId(c),
            );

            match glyph {
                Ok(glyph) => {
                    glyphs.push(Some(glyph));
                }
                Err(err) => {
                    match err {
                        MsdfAtlasLoaderError::GlyphShape(glyph_shape_error) => {
                            glyph_shape_errors.push(glyph_shape_error);
                        }
                        error => return Err(error),
                    }
                    glyphs.push(None);
                }
            }
        }

        let (atlas_size, packed) = Self::pack(&glyphs);
        let texture_size = atlas_size.as_vec2();

        let glyphs: Vec<_> = packed
            .into_iter()
            .zip(glyphs)
            .map(|glyph| {
                if let (Some(position), Some(glyph)) = glyph {
                    let scale = (1.0 / glyph.px_per_em) as f32;
                    let offset = glyph.anchor - 0.5 * scale;

                    let tex_offset = position.as_vec2() / texture_size;
                    let size = Vec2::new(glyph.width as f32, glyph.height as f32) - 1.0;
                    let v1 = Vec2::ZERO;
                    let v2 = Vec2::new(size.x, 0.0);
                    let v3 = Vec2::new(0.0, size.y);
                    let v4 = size;

                    Some(GlyphInfo {
                        position,
                        size: UVec2::new(glyph.width, glyph.height),
                        anchor: offset,
                        shape: glyph,
                        vertices: [
                            GlyphVertex {
                                position: v1 * scale - offset,
                                tex_coords: v1 / texture_size + tex_offset,
                            },
                            GlyphVertex {
                                position: v2 * scale - offset,
                                tex_coords: v2 / texture_size + tex_offset,
                            },
                            GlyphVertex {
                                position: v3 * scale - offset,
                                tex_coords: v3 / texture_size + tex_offset,
                            },
                            GlyphVertex {
                                position: v4 * scale - offset,
                                tex_coords: v4 / texture_size + tex_offset,
                            },
                        ],
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok((
            GlyphAtlas {
                width: atlas_size.x,
                height: atlas_size.y,
                glyphs,
            },
            glyph_shape_errors,
        ))
    }

    fn pack(glyphs: &[Option<GlyphShape>]) -> (UVec2, Vec<Option<UVec2>>) {
        let mut config = rect_packer::Config {
            width: 256,
            height: 256,
            border_padding: 0,
            rectangle_padding: 0,
        };

        let mut packer = Packer::new(config);
        let mut last_switched_width = false;

        let packed = loop {
            let mut out_of_room = false;
            let mut packed = Vec::with_capacity(glyphs.len());

            for glyph in glyphs.iter() {
                let Some(glyph) = glyph else {
                    packed.push(None);
                    continue;
                };

                let Some(rect) = packer.pack(glyph.width as i32, glyph.height as i32, false) else {
                    out_of_room = true;
                    break;
                };

                let position = uvec2(rect.x as u32, rect.y as u32);
                packed.push(Some(position));
            }

            if out_of_room {
                if last_switched_width {
                    config.height *= 2;
                } else {
                    config.width *= 2;
                }

                last_switched_width = !last_switched_width;
                packer = Packer::new(config);
            } else {
                break packed;
            }
        };

        (uvec2(config.width as u32, config.height as u32), packed)
    }

    pub fn generate_full(&self) -> GlyphBitmap {
        let mut bitmap = GlyphBitmap::new(self.width, self.height);

        for glyph in self.glyphs.iter().flatten() {
            let glyph_bitmap = glyph.shape.generate();
            glyph_bitmap.copy_to(&mut bitmap, glyph.position.x, glyph.position.y);
        }

        bitmap
    }
}

pub struct GlyphShape {
    pub anchor: Vec2,
    pub px_per_em: f64,
    pub shape: Shape,
    pub width: u32,
    pub height: u32,
    pub framing: msdfgen::Framing<f64>,
}

impl GlyphShape {
    pub fn new(
        units_per_em: f64,
        px_per_em: f64,
        range: Range<f64>,
        angle_threshold: f64,
        face: &Face,
        glyph: GlyphId,
    ) -> FontResult<Self> {
        let mut shape = face
            .glyph_shape(glyph)
            .ok_or(MsdfAtlasLoaderError::GlyphShape(glyph))?;
        shape.edge_coloring_simple(angle_threshold, 0);
        let bounds = shape.get_bound();
        let px_per_unit = px_per_em / units_per_em;
        let width = (bounds.width() * px_per_unit).ceil() as u32 + 8;
        let height = (bounds.height() * px_per_unit).ceil() as u32 + 8;
        let width = width.max(16);
        let height = height.max(16);
        let framing = bounds.autoframe(width, height, range, None).ok_or(
            MsdfAtlasLoaderError::AutoFraming {
                glyph,
                width: width as usize,
                height: height as usize,
                range,
            },
        )?;

        let anchor =
            Vec2::new(framing.translate.x as f32, framing.translate.y as f32) / units_per_em as f32;

        Ok(Self {
            anchor,
            framing,
            px_per_em,
            shape,
            width,
            height,
        })
    }

    pub fn generate(&self) -> GlyphBitmap {
        let config: MsdfGeneratorConfig = MsdfGeneratorConfig::default();
        let width = self.width;
        let height = self.height;
        let framing = &self.framing;
        let shape = &self.shape;
        let mut bitmap = Bitmap::<Rgba<f32>>::new(width, height);
        shape.generate_mtsdf(&mut bitmap, framing, config);
        shape.correct_sign(&mut bitmap, framing, FillRule::default());
        shape.correct_msdf_error(&mut bitmap, framing, config);

        let data = bitmap
            .pixels()
            .iter()
            .map(|p| {
                fn conv(f: f32) -> u32 {
                    (f * 256.0).round() as u8 as _
                }

                (conv(p.r) << 24) | (conv(p.g) << 16) | (conv(p.b) << 8) | conv(p.a)
            })
            .collect();

        GlyphBitmap {
            data,
            width,
            height,
        }
    }
}

pub struct GlyphBitmap {
    pub data: Vec<u32>,
    pub width: u32,
    pub height: u32,
}

impl GlyphBitmap {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            data: vec![0; (width * height) as usize],
            width,
            height,
        }
    }

    pub fn data_bytes(&self) -> &[u8] {
        unsafe {
            let ptr = self.data.as_ptr();
            let ptr: *const u8 = std::mem::transmute(ptr);
            let len = self.data.len() * 4;
            std::slice::from_raw_parts(ptr, len)
        }
    }

    pub fn copy_to(&self, dst: &mut GlyphBitmap, x: u32, y: u32) {
        if self.width + x > dst.width || self.height + y > dst.height {
            panic!("copy_to out-of-bounds");
        }

        let mut src_cursor = 0;
        let mut dst_cursor = (y * dst.width + x) as usize;
        for _ in 0..self.height {
            let src_range = src_cursor..(src_cursor + self.width as usize);
            let dst_range = dst_cursor..(dst_cursor + self.width as usize);
            dst.data[dst_range].copy_from_slice(&self.data[src_range]);
            src_cursor += self.width as usize;
            dst_cursor += dst.width as usize;
        }
    }
}
