// TODO use common view binding
#import bevy_render::view::View

@group(0) @binding(0) var<uniform> view: View;
@group(1) @binding(0) var<uniform> ubo: Uniform;
@group(2) @binding(0) var atlas_texture: texture_2d<f32>;
@group(2) @binding(1) var atlas_sampler: sampler;
@group(2) @binding(2) var<storage> atlas_glyphs: array<vec2<f32>>;

struct VertexInput {
    @builtin(vertex_index) idx: u32,
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) glyph: u32,
};

struct Uniform {
    transform: mat4x4<f32>,
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
    let position = view.view_proj * ubo.transform * vec4(corner + vertex.position, 0., 1.);
    return VertexOutput(position, vertex.color, uv);
}

fn screen_px_range(tex_coords: vec2<f32>) -> f32 {
    let msdf_range = 8.0;
    let unit_range = vec2<f32>(msdf_range) / vec2<f32>(textureDimensions(atlas_texture, 0));
    let screen_tex_size = vec2<f32>(1.0) / fwidth(tex_coords);
    return max(0.5 * dot(unit_range, screen_tex_size), 1.0);
}

fn median(r: f32, g: f32, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b));
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let msd = textureSample(atlas_texture, atlas_sampler, in.uv);
    let sd = median(msd.r, msd.g, msd.b);
    let dist = screen_px_range(in.uv) * (sd - 0.5);
    let alpha = clamp(dist + 0.5, 0.0, 1.0);
    return vec4<f32>(in.color.rgb, alpha);
}
