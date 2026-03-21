pub mod components;
pub mod config;
pub mod factory;
pub mod utils;

// Re-export main types and functions
pub use config::{
    ComponentType, InferenceState, ModelState, WebComponentConfig, WebComponentFactory,
};
pub use utils::{
    create_web_component_html_template, generate_web_components_package,
    is_web_components_supported,
};
