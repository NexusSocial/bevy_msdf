use bevy::{ecs::system::EntityCommands, prelude::*};
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use bevy_msdf::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin)
        .add_plugins(MsdfPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, ui)
        .init_resource::<Viewer>()
        .run();
}

fn setup(mut commands: Commands, assets: ResMut<AssetServer>) {
    // msdf text
    commands.spawn((
        MsdfText {
            atlas: assets.load("mononoki-Regular.ttf"),
            content: String::new(),
            color: Color::WHITE,
        },
        Transform::default(),
        GlobalTransform::default(),
    ));

    // camera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 0.0, 20.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });
}

fn ui(
    mut contexts: EguiContexts,
    mut viewer: ResMut<Viewer>,
    mut commands: Commands,
    mut text: Query<(Entity, &mut MsdfText)>,
) {
    egui::SidePanel::left("left_panel").show(contexts.ctx_mut(), |ui| {
        viewer.show(ui);
    });

    let (entity, mut text) = text.single_mut();
    viewer.update(&mut commands, entity, &mut text);
}

pub fn show_color(ui: &mut egui::Ui, color: &mut Color) {
    let mut rgba = color.as_rgba_u8();
    ui.color_edit_button_srgba_premultiplied(&mut rgba);
    *color = Color::rgba_u8(rgba[0], rgba[1], rgba[2], rgba[3]);
}

#[derive(Resource)]
struct Viewer {
    contents: String,
    text_color: Color,
    border: OptionalUi<BorderUi>,
    glow: OptionalUi<GlowUi>,
}

impl Default for Viewer {
    fn default() -> Self {
        Self {
            contents: "Hello, world!".to_string(),
            text_color: Color::BLACK,
            border: OptionalUi {
                inner: BorderUi::default(),
                text: "enable border".to_string(),
                show: false,
            },
            glow: OptionalUi {
                inner: GlowUi::default(),
                text: "enable glow".to_string(),
                show: false,
            },
        }
    }
}

impl Viewer {
    fn show(&mut self, ui: &mut egui::Ui) {
        ui.text_edit_multiline(&mut self.contents);
        show_color(ui, &mut self.text_color);
        self.border.show(ui, BorderUi::show);
        self.glow.show(ui, GlowUi::show);
    }

    fn update(&mut self, commands: &mut Commands, entity: Entity, text: &mut MsdfText) {
        text.content.clone_from(&self.contents);
        text.color = self.text_color;

        let mut entity = commands.entity(entity);
        self.border.update(&mut entity, BorderUi::update);
        self.glow.update(&mut entity, GlowUi::update);
    }
}

#[derive(Default)]
struct OptionalUi<T> {
    inner: T,
    text: String,
    show: bool,
}

impl<T> OptionalUi<T> {
    fn show(&mut self, ui: &mut egui::Ui, cb: impl FnOnce(&mut T, &mut egui::Ui)) {
        ui.checkbox(&mut self.show, &self.text);

        if self.show {
            cb(&mut self.inner, ui);
        }
    }

    fn update<C: Component>(&mut self, entity: &mut EntityCommands, cb: impl FnOnce(&T) -> C) {
        if self.show {
            let c = cb(&self.inner);
            entity.insert(c);
        } else {
            entity.remove::<C>();
        }
    }
}

struct BorderUi {
    color: Color,
    size: f32,
}

impl Default for BorderUi {
    fn default() -> Self {
        Self {
            color: Color::WHITE,
            size: 0.20,
        }
    }
}

impl BorderUi {
    fn show(&mut self, ui: &mut egui::Ui) {
        show_color(ui, &mut self.color);
        ui.add(egui::Slider::new(&mut self.size, 0.0..=0.5));
    }

    fn update(&self) -> MsdfBorder {
        MsdfBorder {
            color: self.color,
            size: self.size,
        }
    }
}

struct GlowUi {
    color: Color,
    size: f32,
    offset: Vec2,
}

impl Default for GlowUi {
    fn default() -> Self {
        Self {
            color: Color::BLACK,
            size: 0.4,
            offset: Vec2::new(0.1, 0.1),
        }
    }
}

impl GlowUi {
    fn show(&mut self, ui: &mut egui::Ui) {
        show_color(ui, &mut self.color);
        ui.add(egui::Slider::new(&mut self.size, 0.0..=0.5));
        ui.add(egui::Slider::new(&mut self.offset.x, -1.0..=1.0));
        ui.add(egui::Slider::new(&mut self.offset.y, -1.0..=1.0));
    }

    fn update(&self) -> MsdfGlow {
        MsdfGlow {
            color: self.color,
            size: self.size,
            offset: self.offset,
        }
    }
}
