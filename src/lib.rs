use bevy::{
    asset::{io::Reader, AssetLoader, AsyncReadExt, LoadContext},
    prelude::*,
    utils::BoxedFuture,
};
use owned_ttf_parser::OwnedFace;
use thiserror::Error;

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
            Ok(MsdfAtlas { face })
        })
    }
}

#[derive(Asset, TypePath)]
pub struct MsdfAtlas {
    pub face: OwnedFace,
}

/// A component that draws a list of glyphs onto a plane.
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

pub struct MsdfPlugin;

impl Plugin for MsdfPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<MsdfAtlas>()
            .init_asset_loader::<MsdfAtlasLoader>();
    }
}
