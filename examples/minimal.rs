use bevy::prelude::*;
use bevy_msdf::*;
use owned_ttf_parser::AsFaceRef;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(MsdfPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, layout)
        .run();
}

fn setup(mut commands: Commands, assets: ResMut<AssetServer>) {
    // mononoki MSDF atlas
    let atlas = assets.load("mononoki-Regular.ttf");

    // msdf glyphs
    let glyphs = vec![];
    commands.spawn(MsdfDraw { atlas, glyphs });

    // camera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 0.0, 30.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });
}

fn layout(
    mut atlas_events: EventReader<AssetEvent<MsdfAtlas>>,
    mut draws: Query<&mut MsdfDraw>,
    atlases: Res<Assets<MsdfAtlas>>,
) {
    for ev in atlas_events.read() {
        // ignore everything but load events
        let AssetEvent::LoadedWithDependencies { id } = *ev else {
            continue;
        };

        for mut draw in draws.iter_mut() {
            if id != draw.atlas.id() {
                continue;
            }

            let atlas = atlases.get(id).unwrap();

            let text = "Hello, world!";

            draw.glyphs = text
                .chars()
                .enumerate()
                .filter_map(|(idx, c)| {
                    let pos = Vec2::new(idx as f32, 0.0);
                    let color = Color::WHITE;
                    let index = atlas.face.as_face_ref().glyph_index(c)?.0;
                    Some(MsdfGlyph { pos, color, index })
                })
                .collect();
        }
    }
}
