use bevy::{
    asset::{io::Reader, load_internal_asset, AssetLoader, AsyncReadExt, LoadContext},
    core_pipeline::{
        core_3d::{Transparent3d, CORE_3D_DEPTH_FORMAT},
        prepass::{DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass},
    },
    pbr::{MeshPipeline, MeshPipelineKey, SetMeshViewBindGroup},
    prelude::*,
    render::{
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItem, RenderCommand, RenderCommandResult,
            RenderPhase, SetItemPipeline,
        },
        render_resource::{
            FragmentState, PipelineCache, RenderPipelineDescriptor, SpecializedRenderPipeline,
            SpecializedRenderPipelines, VertexState,
        },
        texture::BevyDefault,
        view::{ExtractedView, RenderLayers, ViewTarget},
        Render, RenderApp, RenderSet,
    },
    utils::BoxedFuture,
};
use owned_ttf_parser::OwnedFace;
use thiserror::Error;
use wgpu::{
    BlendState, ColorTargetState, ColorWrites, CompareFunction, DepthStencilState,
    MultisampleState, PrimitiveState, TextureFormat,
};

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

const SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(167821518087860206701142330598674077861);

pub struct MsdfPlugin;

impl Plugin for MsdfPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, SHADER_HANDLE, "msdf.wgsl", Shader::from_wgsl);

        app.init_asset::<MsdfAtlas>()
            .init_asset_loader::<MsdfAtlasLoader>();

        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_render_command::<Transparent3d, MsdfCommands>()
            .init_resource::<SpecializedRenderPipelines<MsdfPipeline>>()
            .add_systems(Render, queue_msdf_draws.in_set(RenderSet::Queue));
    }

    fn finish(&self, app: &mut App) {
        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.init_resource::<MsdfPipeline>();
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct MsdfPipelineKey {
    view_key: MeshPipelineKey,
}

#[derive(Resource)]
pub struct MsdfPipeline {
    mesh_pipeline: MeshPipeline,
}

impl FromWorld for MsdfPipeline {
    fn from_world(world: &mut World) -> Self {
        Self {
            mesh_pipeline: world.resource::<MeshPipeline>().clone(),
        }
    }
}

impl SpecializedRenderPipeline for MsdfPipeline {
    type Key = MsdfPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let view_layout = self
            .mesh_pipeline
            .get_view_layout(key.view_key.into())
            .clone();

        let format = if key.view_key.contains(MeshPipelineKey::HDR) {
            ViewTarget::TEXTURE_FORMAT_HDR
        } else {
            TextureFormat::bevy_default()
        };

        let layout = vec![view_layout];

        RenderPipelineDescriptor {
            label: Some("MSDF Pipeline".into()),
            layout,
            push_constant_ranges: vec![],
            vertex: VertexState {
                shader: SHADER_HANDLE,
                entry_point: "vertex".into(),
                shader_defs: vec![],
                buffers: vec![],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Greater,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: MultisampleState {
                count: key.view_key.msaa_samples(),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(FragmentState {
                shader: SHADER_HANDLE,
                shader_defs: vec![],
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::all(),
                })],
            }),
        }
    }
}

pub struct DrawMsdf;

impl<P: PhaseItem> RenderCommand<P> for DrawMsdf {
    type Param = ();
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        _view: bevy::ecs::query::ROQueryItem<'w, Self::ViewQuery>,
        _entity: Option<bevy::ecs::query::ROQueryItem<'w, Self::ItemQuery>>,
        _param: bevy::ecs::system::SystemParamItem<'w, '_, Self::Param>,
        pass: &mut bevy::render::render_phase::TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.draw(0..6, 0..1);
        RenderCommandResult::Success
    }
}

type MsdfCommands = (SetItemPipeline, SetMeshViewBindGroup<0>, DrawMsdf);

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn queue_msdf_draws(
    draw_functions: Res<DrawFunctions<Transparent3d>>,
    msaa: Res<Msaa>,
    pipeline: Res<MsdfPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<MsdfPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    mut views: Query<(
        Entity, // temp, used for Transparent3d.entity
        &ExtractedView,
        &mut RenderPhase<Transparent3d>,
        Option<&RenderLayers>,
        (
            Has<NormalPrepass>,
            Has<DepthPrepass>,
            Has<MotionVectorPrepass>,
            Has<DeferredPrepass>,
        ),
    )>,
) {
    let draw_function = draw_functions.read().get_id::<MsdfCommands>().unwrap();

    for (
        entity,
        view,
        mut transparent_phase,
        render_layers,
        (normal_prepass, depth_prepass, motion_vector_prepass, deferred_prepass),
    ) in &mut views
    {
        let _render_layers = render_layers.copied().unwrap_or_default();

        let mut view_key = MeshPipelineKey::from_msaa_samples(msaa.samples())
            | MeshPipelineKey::from_hdr(view.hdr);

        if normal_prepass {
            view_key |= MeshPipelineKey::NORMAL_PREPASS;
        }

        if depth_prepass {
            view_key |= MeshPipelineKey::DEPTH_PREPASS;
        }

        if motion_vector_prepass {
            view_key |= MeshPipelineKey::MOTION_VECTOR_PREPASS;
        }

        if deferred_prepass {
            view_key |= MeshPipelineKey::DEFERRED_PREPASS;
        }

        let key = MsdfPipelineKey { view_key };

        let pipeline = pipelines.specialize(&pipeline_cache, &pipeline, key);

        transparent_phase.add(Transparent3d {
            entity,
            distance: 0.0, // TODO plane distance sorting
            pipeline,
            draw_function,
            batch_range: 0..1,
            dynamic_offset: None,
        });
    }
}
