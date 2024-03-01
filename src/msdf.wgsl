// TODO use common view binding
#import bevy_render::view::View

@group(0) @binding(0) var<uniform> view: View;
@group(1) @binding(0) var<uniform> transform: mat4x4<f32>;
@group(2) @binding(0) var atlas_texture: texture_2d<f32>;
@group(2) @binding(1) var atlas_sampler: sampler;
@group(2) @binding(2) var<storage> atlas_glyphs: array<vec2<f32>>;

struct VertexInput {
    @builtin(vertex_index) idx: u32,
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) glyph: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vertex(vertex: VertexInput) -> VertexOutput {
    let glyph_base = vertex.glyph * 8;
    let corner = atlas_glyphs[glyph_base + vertex.idx];
    let uv = atlas_glyphs[glyph_base + vertex.idx + 4];
    let position = view.view_proj * transform * vec4(corner + vertex.position, 0., 1.);
    return VertexOutput(position, vertex.color, uv);
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
};

@fragment
fn fragment(in: VertexOutput) -> FragmentOutput {
    let tex = textureSample(atlas_texture, atlas_sampler, in.uv);
    return FragmentOutput(in.color * tex);
}
