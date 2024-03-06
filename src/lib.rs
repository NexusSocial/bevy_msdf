use std::sync::Arc;

use bevy::{
    asset::{io::Reader, AssetLoader, AsyncReadExt, LoadContext},
    prelude::*,
    utils::{BoxedFuture, HashSet},
};
use font_mud::{error::FontError, glyph_atlas::GlyphAtlas};
use owned_ttf_parser::{AsFaceRef, OwnedFace};
use thiserror::Error;

pub mod render;

/// Possible errors that can be produced by [MsdfAtlasLoader]
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum MsdfAtlasLoaderError {
    /// An [IO](std::io) Error
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// A [owned_ttf_parser::FaceParsingError] Error
    #[error(transparent)]
    FontInvalid(#[from] owned_ttf_parser::FaceParsingError),
    /// A [font_mud::error::FontError] Error
    // TODO implement Error for FontError upstream
    #[error("font error: {0}")]
    FontError(FontError),
}

impl From<FontError> for MsdfAtlasLoaderError {
    fn from(err: FontError) -> Self {
        MsdfAtlasLoaderError::FontError(err)
    }
}

#[derive(Default)]
pub struct MsdfAtlasLoader;

impl AssetLoader for MsdfAtlasLoader {
    type Asset = MsdfAtlas;
    type Settings = ();
    type Error = MsdfAtlasLoaderError;

    fn load<'a>(
        &'a self,
        reader: &'a mut Reader,
        _settings: &'a (),
        _load_context: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<MsdfAtlas, Self::Error>> {
        Box::pin(async move {
            let mut bytes = Vec::new();
            reader.read_to_end(&mut bytes).await?;
            // TODO support non-zero face indices
            let face = OwnedFace::from_vec(bytes, 0)?;
            let (atlas, _glyph_errors) = GlyphAtlas::new(face.as_face_ref())?;

            Ok(MsdfAtlas {
                face: Arc::new(face),
                atlas: Arc::new(atlas),
            })
        })
    }
}

#[derive(Asset, Clone, TypePath)]
pub struct MsdfAtlas {
    pub face: Arc<OwnedFace>,
    pub atlas: Arc<GlyphAtlas>,
}

/// A bundle of the components necessary to draw a plane of MSDF glyphs.
#[derive(Bundle)]
pub struct MsdfBundle {
    pub draw: MsdfDraw,
    pub transform: Transform,
    pub global_transform: GlobalTransform,
}

/// A component that shapes and draws text using [MsdfDraw].
#[derive(Component)]
pub struct MsdfText {
    /// The [MsdfAtlas] to use for this text.
    pub atlas: Handle<MsdfAtlas>,

    /// The text to render.
    pub content: String,

    /// The text's color.
    pub color: Color,
}

/// A component that draws a list of atlas glyphs onto a plane.
#[derive(Component)]
pub struct MsdfDraw {
    /// The [MsdfAtlas] to use for this draw.
    pub atlas: Handle<MsdfAtlas>,

    /// The list of glyphs to draw.
    pub glyphs: Vec<MsdfGlyph>,
}

/// A single instance of a MSDF glyph.
pub struct MsdfGlyph {
    /// The position of this glyph's anchor.
    pub pos: Vec2,

    /// The color to draw this glyph.
    pub color: Color,

    /// The index of this glyph within the [MsdfAtlas].
    pub index: u16,
}

pub fn layout(
    mut commands: Commands,
    mut atlas_events: EventReader<AssetEvent<MsdfAtlas>>,
    texts: Query<(Entity, Ref<MsdfText>)>,
    atlases: Res<Assets<MsdfAtlas>>,
) {
    let loaded_atlases = atlas_events
        .read()
        .filter_map(|ev| match ev {
            AssetEvent::LoadedWithDependencies { id } => Some(id),
            _ => None,
        })
        .collect::<HashSet<_>>();

    for (entity, text) in texts.iter() {
        if !text.is_changed() && !loaded_atlases.contains(&text.atlas.id()) {
            continue;
        }

        let Some(atlas) = atlases.get(&text.atlas) else {
            continue;
        };

        let mut draw = MsdfDraw {
            atlas: text.atlas.clone(),
            glyphs: vec![],
        };

        let mut cursor = 0.0;

        for c in text.content.chars() {
            if let Some(glyph) = atlas.face.as_face_ref().glyph_index(c) {
                draw.glyphs.push(MsdfGlyph {
                    pos: Vec2::new(cursor, 0.0),
                    color: text.color,
                    index: glyph.0,
                });
            }

            cursor += 0.7;
        }

        draw.glyphs
            .iter_mut()
            .for_each(|glyph| glyph.pos.x -= cursor / 2.0);

        commands.entity(entity).insert(draw);
    }
}

pub struct MsdfPlugin;

impl Plugin for MsdfPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<MsdfAtlas>()
            .init_asset_loader::<MsdfAtlasLoader>()
            .add_plugins(render::MsdfRenderPlugin)
            .add_systems(PostUpdate, layout);
    }
}
