use std::{ops::Range, sync::Arc};

use bevy::{
    asset::load_internal_asset,
    core_pipeline::{
        core_3d::{Transparent3d, CORE_3D_DEPTH_FORMAT},
        prepass::{DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass},
    },
    ecs::system::{lifetimeless::SRes, SystemParam},
    pbr::{MeshPipeline, MeshPipelineKey, SetMeshViewBindGroup},
    prelude::*,
    render::{
        render_asset::{
            PrepareAssetError, RenderAsset, RenderAssetPlugin, RenderAssetUsages, RenderAssets,
        },
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItem, RenderCommand, RenderCommandResult,
            RenderPhase, SetItemPipeline,
        },
        render_resource::{
            binding_types::{sampler, storage_buffer_read_only_sized, texture_2d, uniform_buffer},
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, BufferVec,
            FragmentState, PipelineCache, RenderPipelineDescriptor, ShaderType,
            SpecializedRenderPipeline, SpecializedRenderPipelines, Texture, VertexBufferLayout,
            VertexState,
        },
        renderer::{RenderDevice, RenderQueue},
        texture::BevyDefault,
        view::{ExtractedView, RenderLayers, ViewTarget},
        Extract, Render, RenderApp, RenderSet,
    },
    tasks::{block_on, futures_lite::future, AsyncComputeTaskPool, Task},
    utils::{HashMap, HashSet},
};
use bytemuck::{Pod, Zeroable};
use font_mud::{glyph_atlas::GlyphAtlas, glyph_bitmap::GlyphBitmap};
use wgpu::{util::TextureDataOrder, *};

use crate::{MsdfAtlas, MsdfDraw};

const SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(167821518087860206701142330598674077861);

/// The prepared [MsdfAtlas] for rendering.
#[derive(Asset, TypePath)]
pub struct GpuMsdfAtlas {
    pub atlas: Arc<GlyphAtlas>,
    pub texture: Texture,
    pub has_glyphs: HashSet<u16>,
    pub bind_group: BindGroup,
}

impl RenderAsset for MsdfAtlas {
    type PreparedAsset = GpuMsdfAtlas;
    type Param = (SRes<RenderDevice>, SRes<RenderQueue>, SRes<MsdfPipeline>);

    fn asset_usage(&self) -> RenderAssetUsages {
        RenderAssetUsages::RENDER_WORLD
    }

    fn prepare_asset(
        self,
        (device, queue, pipeline): &mut <Self::Param as SystemParam>::Item<'_, '_>,
    ) -> Result<Self::PreparedAsset, PrepareAssetError<Self>> {
        let format = TextureFormat::Rgba8Unorm;

        let texture_data = vec![0; (self.atlas.width * self.atlas.height) as usize * 4];

        let texture = device.create_texture_with_data(
            queue,
            &TextureDescriptor {
                label: Some("msdf atlas"),
                size: Extent3d {
                    width: self.atlas.width,
                    height: self.atlas.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
                view_formats: &[format],
            },
            TextureDataOrder::LayerMajor,
            &texture_data,
        );

        let texture_view = texture.create_view(&Default::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("msdf atlas sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let glyph_data: Vec<_> = self
            .atlas
            .glyphs
            .iter()
            .map(|glyph| {
                glyph.as_ref().map(|info| GpuMsdfGlyphSource {
                    positions: info.vertices.map(|v| v.position.to_array().into()),
                    tex_coords: info.vertices.map(|v| v.tex_coords.to_array().into()),
                })
            })
            .map(|glyph| {
                glyph.unwrap_or(GpuMsdfGlyphSource {
                    positions: [Vec2::ZERO; 4],
                    tex_coords: [Vec2::ZERO; 4],
                })
            })
            .collect();

        let glyphs = device.create_buffer_with_data(&wgpu::util::BufferInitDescriptor {
            label: Some("msdf atlas glyphs"),
            contents: bytemuck::cast_slice(&glyph_data),
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
        });

        let bind_group = device.create_bind_group(
            "msdf atlas bind group",
            &pipeline.atlas_bgl,
            &BindGroupEntries::sequential((&texture_view, &sampler, glyphs.as_entire_binding())),
        );

        Ok(GpuMsdfAtlas {
            atlas: self.atlas.to_owned(),
            texture,
            has_glyphs: HashSet::new(),
            bind_group,
        })
    }
}

/// A plugin for the Bevy render subapp implementing MSDF rendering.
pub struct MsdfRenderPlugin;

impl Plugin for MsdfRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(RenderAssetPlugin::<MsdfAtlas>::default());

        load_internal_asset!(app, SHADER_HANDLE, "msdf.wgsl", Shader::from_wgsl);

        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_render_command::<Transparent3d, MsdfCommands>()
            .init_resource::<SpecializedRenderPipelines<MsdfPipeline>>()
            .init_resource::<MsdfBuffers>()
            .init_resource::<MsdfBindGroups>()
            .add_systems(
                Render,
                (
                    flush_atlas_writes.in_set(RenderSet::Prepare),
                    prepare_msdf_resources.in_set(RenderSet::PrepareResources),
                    prepare_msdf_bind_groups.in_set(RenderSet::PrepareBindGroups),
                    queue_msdf_draws.in_set(RenderSet::Queue),
                ),
            )
            .add_systems(ExtractSchedule, extract_msdfs);
    }

    fn finish(&self, app: &mut App) {
        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.init_resource::<MsdfPipeline>();
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuMsdfGlyph {
    pub position: Vec2,
    pub color: u32,
    pub index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, ShaderType)]
pub struct GpuUniform {
    pub transform: Mat4,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuMsdfGlyphSource {
    pub positions: [Vec2; 4],
    pub tex_coords: [Vec2; 4],
}

#[derive(Component)]
pub struct RenderMsdfDraw {
    pub atlas: AssetId<MsdfAtlas>,
    pub vertices: Range<u32>,
    pub position: Vec3,
    pub uniform: usize,
}

pub struct MsdfAtlasWrite {
    pub atlas: AssetId<MsdfAtlas>,
    pub position: UVec2,
    pub size: UVec2,
    pub task: Task<GlyphBitmap>,
}

#[derive(Resource)]
pub struct MsdfBuffers {
    pub glyphs: BufferVec<GpuMsdfGlyph>,
    pub uniforms: BufferVec<GpuUniform>,
    pub pending_writes: Vec<MsdfAtlasWrite>,
}

impl Default for MsdfBuffers {
    fn default() -> Self {
        Self {
            glyphs: BufferVec::new(BufferUsages::COPY_DST | BufferUsages::VERTEX),
            uniforms: BufferVec::new(BufferUsages::COPY_DST | BufferUsages::UNIFORM),
            pending_writes: vec![],
        }
    }
}

#[derive(Default, Resource)]
pub struct MsdfBindGroups {
    pub uniforms: Option<BindGroup>,
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct MsdfPipelineKey {
    view_key: MeshPipelineKey,
}

#[derive(Resource)]
pub struct MsdfPipeline {
    pub mesh_pipeline: MeshPipeline,
    pub uniforms_bgl: BindGroupLayout,
    pub atlas_bgl: BindGroupLayout,
}

impl FromWorld for MsdfPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let uniforms_bgl = render_device.create_bind_group_layout(
            Some("msdf_uniforms_bind_group_layout"),
            &BindGroupLayoutEntries::single(
                ShaderStages::VERTEX,
                uniform_buffer::<GpuUniform>(true),
            ),
        );

        let atlas_bgl = render_device.create_bind_group_layout(
            Some("msdf_atlas_bind_group_layout"),
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    sampler(SamplerBindingType::Filtering),
                    storage_buffer_read_only_sized(false, None),
                ),
            ),
        );

        Self {
            mesh_pipeline: world.resource::<MeshPipeline>().clone(),
            uniforms_bgl,
            atlas_bgl,
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

        let layout = vec![
            view_layout,
            self.uniforms_bgl.clone(),
            self.atlas_bgl.clone(),
        ];

        let vertex_layout = VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuMsdfGlyph>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: vec![
                VertexAttribute {
                    format: VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
                VertexAttribute {
                    format: VertexFormat::Unorm8x4,
                    offset: 8,
                    shader_location: 1,
                },
                VertexAttribute {
                    format: VertexFormat::Uint32,
                    offset: 12,
                    shader_location: 2,
                },
            ],
        };

        RenderPipelineDescriptor {
            label: Some("MSDF Pipeline".into()),
            layout,
            push_constant_ranges: vec![],
            vertex: VertexState {
                shader: SHADER_HANDLE,
                entry_point: "vertex".into(),
                shader_defs: vec![],
                buffers: vec![vertex_layout],
            },
            primitive: PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..default()
            },
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
    type Param = (
        SRes<MsdfBuffers>,
        SRes<MsdfBindGroups>,
        SRes<RenderAssets<MsdfAtlas>>,
    );

    type ViewQuery = ();
    type ItemQuery = &'static RenderMsdfDraw;

    fn render<'w>(
        _item: &P,
        _view: bevy::ecs::query::ROQueryItem<'w, Self::ViewQuery>,
        entity: Option<bevy::ecs::query::ROQueryItem<'w, Self::ItemQuery>>,
        (buffers, bind_groups, atlases): bevy::ecs::system::SystemParamItem<'w, '_, Self::Param>,
        pass: &mut bevy::render::render_phase::TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(draw) = entity else {
            return RenderCommandResult::Failure;
        };

        let Some(atlas) = atlases.into_inner().get(draw.atlas) else {
            return RenderCommandResult::Failure;
        };

        let Some(instances) = buffers.into_inner().glyphs.buffer() else {
            return RenderCommandResult::Failure;
        };

        let Some(uniforms) = bind_groups.into_inner().uniforms.as_ref() else {
            return RenderCommandResult::Failure;
        };

        let uniforms_offset = (draw.uniform * std::mem::size_of::<Mat4>()) as u32;

        pass.set_vertex_buffer(0, instances.slice(..));
        pass.set_bind_group(1, uniforms, &[uniforms_offset]);
        pass.set_bind_group(2, &atlas.bind_group, &[]);
        pass.draw(0..4, draw.vertices.clone());

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
    draws: Query<(Entity, &RenderMsdfDraw)>,
    mut views: Query<(
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

        let eye = view.transform.translation();

        for (entity, draw) in draws.iter() {
            // TODO ensure this actually works
            let distance = eye.distance(draw.position);

            transparent_phase.add(Transparent3d {
                entity,
                distance,
                pipeline,
                draw_function,
                batch_range: 0..1,
                dynamic_offset: None,
            });
        }
    }
}

pub fn extract_msdfs(
    mut commands: Commands,
    in_msdfs: Extract<Query<(&MsdfDraw, &GlobalTransform)>>,
    mut out_msdfs: ResMut<MsdfBuffers>,
    mut atlases: ResMut<RenderAssets<MsdfAtlas>>,
) {
    out_msdfs.glyphs.clear();
    out_msdfs.uniforms.clear();

    let mut used_glyphs: HashMap<AssetId<MsdfAtlas>, HashSet<u16>> = HashMap::new();

    for (msdf, transform) in in_msdfs.iter() {
        let position = transform.translation();

        let uniform = out_msdfs.uniforms.push(GpuUniform {
            transform: transform.compute_matrix(),
        });

        let start = out_msdfs.glyphs.len() as u32;

        out_msdfs
            .glyphs
            .extend(msdf.glyphs.iter().map(|glyph| GpuMsdfGlyph {
                position: glyph.pos,
                color: glyph.color.as_rgba_u32(),
                index: glyph.index as u32,
            }));

        let end = out_msdfs.glyphs.len() as u32;

        commands.spawn(RenderMsdfDraw {
            atlas: msdf.atlas.id(),
            vertices: start..end,
            position,
            uniform,
        });

        used_glyphs
            .entry(msdf.atlas.id())
            .or_default()
            .extend(msdf.glyphs.iter().map(|glyph| glyph.index));
    }

    let thread_pool = AsyncComputeTaskPool::get();

    for (atlas_id, used_glyphs) in used_glyphs {
        let Some(atlas) = atlases.get_mut(atlas_id) else {
            continue;
        };

        for glyph in used_glyphs {
            if !atlas.has_glyphs.insert(glyph) {
                // atlas already has this glyph; skip write queuing
                continue;
            }

            let Some(Some(info)) = atlas.atlas.glyphs.get(glyph as usize) else {
                // atlas does not contain glyph; skip write queuing
                continue;
            };

            let task = thread_pool.spawn({
                // TODO derive Clone on GlyphShape upstream
                let atlas = atlas.atlas.clone();
                async move {
                    let info = atlas.glyphs.get(glyph as usize).unwrap().as_ref().unwrap();
                    info.shape.generate()
                }
            });

            // queue up the atlas write
            out_msdfs.pending_writes.push(MsdfAtlasWrite {
                atlas: atlas_id,
                position: info.position.to_array().into(),
                size: info.size.to_array().into(),
                task,
            });
        }
    }
}

pub fn flush_atlas_writes(
    queue: Res<RenderQueue>,
    atlases: ResMut<RenderAssets<MsdfAtlas>>,
    mut bufs: ResMut<MsdfBuffers>,
) {
    bufs.pending_writes.retain_mut(|write| {
        let Some(bitmap) = block_on(future::poll_once(&mut write.task)) else {
            // not done; wait until next frame.
            return true;
        };

        let Some(atlas) = atlases.get(write.atlas) else {
            // asset has been freed; throw away results
            return false;
        };

        // queue up the uploading of the glyph data
        queue.0.write_texture(
            wgpu::ImageCopyTextureBase {
                texture: &atlas.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: write.position.x,
                    y: write.position.y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            bitmap.data_bytes(),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bitmap.width * 4),
                rows_per_image: Some(bitmap.height),
            },
            Extent3d {
                width: write.size.x,
                height: write.size.y,
                depth_or_array_layers: 1,
            },
        );

        false
    });
}

pub fn prepare_msdf_resources(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut bufs: ResMut<MsdfBuffers>,
) {
    bufs.glyphs.write_buffer(&device, &queue);
    bufs.uniforms.write_buffer(&device, &queue);
}

pub fn prepare_msdf_bind_groups(
    device: Res<RenderDevice>,
    pipeline: Res<MsdfPipeline>,
    bufs: Res<MsdfBuffers>,
    mut groups: ResMut<MsdfBindGroups>,
) {
    groups.uniforms = None;

    if let Some(uniforms) = bufs.uniforms.buffer() {
        groups.uniforms = Some(device.create_bind_group(
            "msdf_uniforms_bind_group",
            &pipeline.uniforms_bgl,
            &BindGroupEntries::single(uniforms.as_entire_binding()),
        ));
    }
}
