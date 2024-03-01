// TODO use common view binding
#import bevy_render::view::View

@group(0) @binding(0) var<uniform> view: View;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vertex(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var positions = array<vec3<f32>, 6>(
        vec3(-0.5, 0., 0.),
        vec3(-0.5, 1., 0.),
        vec3(0.5, 1., 0.),
        vec3(-0.5, 0., 0.),
        vec3(0.5, 1., 0.),
        vec3(0.5, 0, 0.),
    );

    let position = view.view_proj * vec4(positions[idx], 1.);
    let color = vec4<f32>(1.0, 0.0, 1.0, 1.0);
    return VertexOutput(position, color);
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
