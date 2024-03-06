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
    border_color: vec4<f32>,
    glow_color: vec4<f32>,
    glow_offset_size: vec3<f32>,
    border_size: f32,
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

fn screen_px_range(unit_range: vec2<f32>, uv: vec2<f32>) -> f32 {
    let screen_tex_size = vec2<f32>(1.0) / fwidth(uv);
    return max(0.5 * dot(unit_range, screen_tex_size), 1.0);
}

fn median(r: f32, g: f32, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b));
}

fn msdf_alpha_at(unit_range: vec2<f32>, uv: vec2<f32>, bias: f32) -> f32 {
    let msd = textureSample(atlas_texture, atlas_sampler, uv);
    let sd = median(msd.r, msd.g, msd.b);
    let dist = screen_px_range(unit_range, uv) * (sd - bias);
    return clamp(dist + 0.5, 0.0, 1.0);
}

fn blend(base: vec4<f32>, layer: vec3<f32>, blend: f32) -> vec4<f32> {
    return base * (1.0 - blend) + vec4<f32>(layer, blend) * blend;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // calculate unit range for repeat screen coverage calculations
    let msdf_range = 8.0;
    let unit_range = vec2<f32>(msdf_range) / vec2<f32>(textureDimensions(atlas_texture, 0));

    // begin blending together colors
    var output = vec4<f32>(0.0);

    // glow
    if ubo.glow_offset_size.x > 0.0 {
        let uv = in.uv + ubo.glow_offset_size.xy;
        let glow = msdf_alpha_at(unit_range, uv, 0.5 - ubo.glow_offset_size.z);
        output = blend(output, ubo.glow_color.rgb, glow * ubo.glow_color.a);
    }

    // border
    if ubo.border_size > 0.0 {
        let border = msdf_alpha_at(unit_range, in.uv, 0.5 - ubo.border_size);
        output = blend(output, ubo.border_color.rgb, border * ubo.border_color.a);
    }

    // main color
    let main = msdf_alpha_at(unit_range, in.uv, 0.5);
    output = blend(output, in.color.rgb, main * in.color.a);

    return output;
}
