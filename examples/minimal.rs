use bevy::prelude::*;
use bevy_msdf::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(MsdfPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, (layout, rotate))
        .run();
}

fn setup(mut commands: Commands, assets: ResMut<AssetServer>) {
    // msdf text
    commands.spawn((
        MsdfText {
            atlas: assets.load("mononoki-Regular.ttf"),
            content: "Hello, world!".to_string(),
            color: Color::WHITE,
        },
        Transform::from_rotation(Quat::from_rotation_z(1.0)),
        GlobalTransform::default(),
    ));

    // camera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 0.0, 20.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });
}

fn rotate(mut texts: Query<&mut Transform, With<MsdfText>>) {
    for mut text in texts.iter_mut() {
        text.rotate_z(0.01);
    }
}
