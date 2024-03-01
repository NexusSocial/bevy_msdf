// TODO use common view binding
#import bevy_render::view::View

@group(0) @binding(0) var<uniform> view: View;

struct VertexInput {
    @builtin(vertex_index) idx: u32,
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) glyph: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vertex(vertex: VertexInput) -> VertexOutput {
    var corners = array<vec2<f32>, 6>(
        vec2(-0.5, 0.),
        vec2(-0.5, 1.),
        vec2(0.5, 1.),
        vec2(-0.5, 0.),
        vec2(0.5, 1.),
        vec2(0.5, 0.),
    );

    let corner = corners[vertex.idx] * 0.1;
    let position = view.view_proj * vec4(corner + vertex.position, 0., 1.);
    return VertexOutput(position, vertex.color);
}

struct FragmentInput {
    @location(0) color: vec4<f32>,
};

struct FragmentOutput {
    @location(0) color: vec4<f32>,
};

@fragment
fn fragment(in: FragmentInput) -> FragmentOutput {
    return FragmentOutput(in.color);
}
