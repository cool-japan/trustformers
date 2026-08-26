//! Plugin loading infrastructure.

use crate::errors::{Result, TrustformersError};
use crate::plugins::{Plugin, PluginInfo};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Plugin loader for dynamic loading and instantiation.
///
/// The `PluginLoader` handles the runtime loading of plugin libraries,
/// symbol resolution, and plugin instantiation. It supports various
/// plugin formats and provides caching for performance.
///
/// # Supported Formats
///
/// - Dynamic libraries (.so, .dll, .dylib)
/// - WebAssembly modules (.wasm)
/// - Static plugins (compiled-in)
///
/// # Example
///
/// ```no_run
/// use trustformers_core::plugins::{PluginLoader, PluginInfo};
/// use std::path::Path;
///
/// let loader = PluginLoader::new();
///
/// // Load plugin info from metadata
/// let info = loader.load_plugin_info(Path::new("plugins/custom_attention.so")).unwrap();
///
/// // Load the actual plugin
/// let plugin = loader.load_plugin(&info).unwrap();
/// ```
#[derive(Debug)]
pub struct PluginLoader {
    /// Cache of loaded libraries
    library_cache: Arc<Mutex<HashMap<String, LibraryHandle>>>,
    /// Static plugin registry
    static_plugins: Arc<Mutex<HashMap<String, StaticPluginFactory>>>,
    /// Cache hit counter
    cache_hits: Arc<Mutex<u64>>,
    /// Cache miss counter
    cache_misses: Arc<Mutex<u64>>,
    /// Loader configuration
    #[allow(dead_code)]
    config: LoaderConfig,
}

impl PluginLoader {
    /// Creates a new plugin loader.
    ///
    /// # Returns
    ///
    /// A new loader instance with default configuration.
    pub fn new() -> Self {
        Self {
            library_cache: Arc::new(Mutex::new(HashMap::new())),
            static_plugins: Arc::new(Mutex::new(HashMap::new())),
            cache_hits: Arc::new(Mutex::new(0)),
            cache_misses: Arc::new(Mutex::new(0)),
            config: LoaderConfig::default(),
        }
    }

    /// Creates a plugin loader with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Loader configuration
    ///
    /// # Returns
    ///
    /// A new loader instance.
    pub fn with_config(config: LoaderConfig) -> Self {
        Self {
            library_cache: Arc::new(Mutex::new(HashMap::new())),
            static_plugins: Arc::new(Mutex::new(HashMap::new())),
            cache_hits: Arc::new(Mutex::new(0)),
            cache_misses: Arc::new(Mutex::new(0)),
            config,
        }
    }

    /// Loads plugin information from a file or metadata.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the plugin file or metadata
    ///
    /// # Returns
    ///
    /// Plugin information if successfully loaded.
    ///
    /// # Errors
    ///
    /// - File not found
    /// - Invalid plugin format
    /// - Metadata parsing errors
    pub fn load_plugin_info<P: AsRef<Path>>(&self, path: P) -> Result<PluginInfo> {
        let path = path.as_ref();

        // Check for companion metadata file
        let metadata_path = path.with_extension("json");
        if metadata_path.exists() {
            return self.load_metadata_file(&metadata_path);
        }

        // Try to load embedded metadata from the plugin file
        self.load_embedded_metadata(path)
    }

    /// Loads a plugin instance from plugin information.
    ///
    /// # Arguments
    ///
    /// * `info` - Plugin information containing loading details
    ///
    /// # Returns
    ///
    /// A boxed plugin instance ready for use.
    ///
    /// # Errors
    ///
    /// - Plugin file not found
    /// - Symbol resolution failures
    /// - Plugin initialization errors
    pub fn load_plugin(&self, info: &PluginInfo) -> Result<Box<dyn Plugin>> {
        // Check if it's a static plugin first
        if let Ok(static_plugins) = self.static_plugins.lock() {
            if let Some(factory) = static_plugins.get(info.name()) {
                return factory();
            }
        }

        // Load as dynamic library
        self.load_dynamic_plugin(info)
    }

    /// Registers a static plugin factory.
    ///
    /// Static plugins are compiled into the binary and don't require
    /// dynamic loading. This method registers a factory function
    /// that can create instances of the plugin.
    ///
    /// # Arguments
    ///
    /// * `name` - Plugin name
    /// * `factory` - Factory function for creating plugin instances
    ///
    /// # Returns
    ///
    /// `Ok(())` on successful registration.
    pub fn register_static_plugin(&self, name: &str, factory: StaticPluginFactory) -> Result<()> {
        let mut static_plugins = self
            .static_plugins
            .lock()
            .map_err(|_| TrustformersError::lock_error("Failed to acquire lock".to_string()))?;

        static_plugins.insert(name.to_string(), factory);
        Ok(())
    }

    /// Unloads a plugin library from the cache.
    ///
    /// # Arguments
    ///
    /// * `name` - Plugin name to unload
    ///
    /// # Returns
    ///
    /// `Ok(())` on success.
    pub fn unload_library(&self, name: &str) -> Result<()> {
        let mut cache = self
            .library_cache
            .lock()
            .map_err(|_| TrustformersError::lock_error("Failed to acquire lock".to_string()))?;

        cache.remove(name);
        Ok(())
    }

    /// Clears all cached libraries.
    pub fn clear_cache(&self) -> Result<()> {
        let mut cache = self
            .library_cache
            .lock()
            .map_err(|_| TrustformersError::lock_error("Failed to acquire lock".to_string()))?;

        cache.clear();
        Ok(())
    }

    /// Gets loader statistics.
    ///
    /// # Returns
    ///
    /// Loader statistics including cache information.
    pub fn stats(&self) -> Result<LoaderStats> {
        let cache = self
            .library_cache
            .lock()
            .map_err(|_| TrustformersError::lock_error("Failed to acquire lock".to_string()))?;
        let static_plugins = self
            .static_plugins
            .lock()
            .map_err(|_| TrustformersError::lock_error("Failed to acquire lock".to_string()))?;

        let cache_hits = self
            .cache_hits
            .lock()
            .map_err(|_| TrustformersError::lock_error("Failed to acquire lock".to_string()))?;
        let cache_misses = self
            .cache_misses
            .lock()
            .map_err(|_| TrustformersError::lock_error("Failed to acquire lock".to_string()))?;

        Ok(LoaderStats {
            cached_libraries: cache.len(),
            static_plugins: static_plugins.len(),
            cache_hits: *cache_hits,
            cache_misses: *cache_misses,
        })
    }

    /// Loads metadata from a JSON file.
    fn load_metadata_file<P: AsRef<Path>>(&self, path: P) -> Result<PluginInfo> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| TrustformersError::io_error(format!("Failed to read metadata: {}", e)))?;

        serde_json::from_str(&content)
            .map_err(|e| TrustformersError::serialization_error(format!("Invalid metadata: {}", e)))
    }

    /// Read plugin metadata embedded in the plugin file itself.
    ///
    /// Not supported: this crate defines no embedded-metadata section and
    /// cannot read one out of an arbitrary shared object. Inventing
    /// `PluginInfo::new(filename, "1.0.0", ...)` would report a version and
    /// dependency set that were never declared, which downstream
    /// version/dependency checks would then trust. Ship a companion
    /// `<plugin>.json` describing the plugin instead.
    fn load_embedded_metadata<P: AsRef<Path>>(&self, path: P) -> Result<PluginInfo> {
        let path = path.as_ref();
        Err(TrustformersError::plugin_error(format!(
            "{} has no companion metadata file and this loader cannot read embedded metadata; \
             provide {}.json describing the plugin (name, version, description, dependencies)",
            path.display(),
            path.with_extension("").display()
        )))
    }

    /// Loads a plugin as a dynamic library.
    fn load_dynamic_plugin(&self, info: &PluginInfo) -> Result<Box<dyn Plugin>> {
        // Check cache first
        {
            let cache = self
                .library_cache
                .lock()
                .map_err(|_| TrustformersError::lock_error("Failed to acquire lock".to_string()))?;

            if let Some(handle) = cache.get(info.name()) {
                // Increment cache hit counter
                if let Ok(mut hits) = self.cache_hits.lock() {
                    *hits += 1;
                }
                return handle.create_plugin();
            }
        }

        // Cache miss - increment counter
        if let Ok(mut misses) = self.cache_misses.lock() {
            *misses += 1;
        }

        // Load the library
        let handle = LibraryHandle::load(info)?;
        let plugin = handle.create_plugin()?;

        // Cache the handle
        {
            let mut cache = self
                .library_cache
                .lock()
                .map_err(|_| TrustformersError::lock_error("Failed to acquire lock".to_string()))?;
            cache.insert(info.name().to_string(), handle);
        }

        Ok(plugin)
    }
}

impl Default for PluginLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Type alias for static plugin factory functions.
pub type StaticPluginFactory = fn() -> Result<Box<dyn Plugin>>;

/// Handle to a loaded dynamic library.
///
/// This struct manages the lifetime of a loaded plugin library
/// and provides symbol resolution for plugin creation.
#[derive(Debug)]
struct LibraryHandle {
    /// Library name
    #[allow(dead_code)]
    name: String,
    /// Entry point information
    _entry_point: String,
}

impl LibraryHandle {
    /// Loads a plugin library.
    ///
    /// # Arguments
    ///
    /// * `info` - Plugin information
    ///
    /// # Returns
    ///
    /// A library handle if loading succeeds.
    /// Load a plugin's shared library.
    ///
    /// Not implemented: `trustformers-core` links no dynamic loader, so no
    /// library can be opened. Returning a handle for a library that was never
    /// opened — and then caching it — meant `unload_library` reported success
    /// for a library that never existed.
    ///
    /// Register plugins statically with
    /// [`PluginLoader::register_static_plugin`] instead.
    fn load(info: &PluginInfo) -> Result<Self> {
        Err(TrustformersError::plugin_error(format!(
            "cannot load plugin '{}' from {}: dynamic library loading is not implemented in \
             trustformers-core. Register the plugin with \
             PluginLoader::register_static_plugin instead.",
            info.name(),
            info.entry_point()
        )))
    }

    /// Creates a plugin instance from this library.
    ///
    /// # Returns
    ///
    /// A boxed plugin instance.
    /// Instantiate the plugin from its loaded library.
    ///
    /// Unreachable while [`Self::load`] refuses to open a library; kept so the
    /// symbol-resolution step has a home when a loader is wired up.
    fn create_plugin(&self) -> Result<Box<dyn Plugin>> {
        Err(TrustformersError::plugin_error(format!(
            "cannot instantiate plugin '{}': dynamic symbol resolution is not implemented",
            self.name
        )))
    }
}

/// Plugin loader configuration.
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    /// Enable library caching
    pub cache_enabled: bool,
    /// Maximum number of cached libraries
    pub max_cached_libraries: usize,
    /// Plugin load timeout in seconds
    pub load_timeout_secs: u64,
    /// Enable lazy loading
    pub lazy_loading: bool,
    /// Symbol name prefix for plugin factories
    pub symbol_prefix: String,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            cache_enabled: true,
            max_cached_libraries: 50,
            load_timeout_secs: 30,
            lazy_loading: true,
            symbol_prefix: "create_plugin".to_string(),
        }
    }
}

/// Plugin loader statistics.
#[derive(Debug, Clone)]
pub struct LoaderStats {
    /// Number of cached libraries
    pub cached_libraries: usize,
    /// Number of registered static plugins
    pub static_plugins: usize,
    /// Cache hit count
    pub cache_hits: u64,
    /// Cache miss count
    pub cache_misses: u64,
}

/// Plugin loading error types.
#[derive(Debug, Clone)]
pub enum LoadError {
    /// Library file not found
    LibraryNotFound(String),
    /// Symbol not found in library
    SymbolNotFound(String),
    /// Plugin initialization failed
    InitializationFailed(String),
    /// Invalid plugin format
    InvalidFormat(String),
    /// Version incompatibility
    VersionMismatch(String),
    /// Dependency not satisfied
    DependencyNotSatisfied(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::LibraryNotFound(path) => write!(f, "Library not found: {}", path),
            LoadError::SymbolNotFound(symbol) => write!(f, "Symbol not found: {}", symbol),
            LoadError::InitializationFailed(msg) => write!(f, "Initialization failed: {}", msg),
            LoadError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            LoadError::VersionMismatch(msg) => write!(f, "Version mismatch: {}", msg),
            LoadError::DependencyNotSatisfied(dep) => {
                write!(f, "Dependency not satisfied: {}", dep)
            },
        }
    }
}

impl std::error::Error for LoadError {}

/// Macro for registering static plugins.
///
/// This macro generates the boilerplate code needed to register
/// a static plugin with the loader.
///
/// # Example
///
/// ```no_run
/// use trustformers_core::register_static_plugin;
/// use trustformers_core::plugins::Plugin;
/// use trustformers_core::tensor::Tensor;
/// use trustformers_core::errors::Result;
/// use std::collections::HashMap;
///
/// #[derive(Debug, Clone, Default)]
/// struct MyPlugin {
///     config: HashMap<String, serde_json::Value>,
/// }
/// impl Plugin for MyPlugin {
///     fn name(&self) -> &str { "my_plugin" }
///     fn version(&self) -> &str { "1.0.0" }
///     fn description(&self) -> &str { "My custom plugin" }
///     fn configure(&mut self, config: HashMap<String, serde_json::Value>) -> Result<()> {
///         self.config = config; Ok(())
///     }
///     fn get_config(&self) -> &HashMap<String, serde_json::Value> { &self.config }
///     fn as_any(&self) -> &dyn std::any::Any { self }
///     fn forward(&self, input: Tensor) -> Result<Tensor> { Ok(input) }
/// }
///
/// register_static_plugin!(MyPlugin, "my_plugin");
/// ```
#[macro_export]
macro_rules! register_static_plugin {
    ($plugin_type:ty, $name:expr) => {
        pub fn register_plugin() -> $crate::errors::Result<Box<dyn $crate::plugins::Plugin>> {
            Ok(Box::new(<$plugin_type>::default()))
        }

        #[cfg(feature = "static-plugins")]
        #[ctor::ctor]
        fn register() {
            use $crate::plugins::PluginLoader;

            let loader = PluginLoader::new();
            let _ = loader.register_static_plugin($name, register_plugin);
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal plugin used to prove the static path still works.
    #[derive(Debug, Default, Clone)]
    struct StaticTestPlugin {
        config: HashMap<String, serde_json::Value>,
    }

    impl Plugin for StaticTestPlugin {
        fn name(&self) -> &str {
            "static-one"
        }

        fn version(&self) -> &str {
            "1.0.0"
        }

        fn description(&self) -> &str {
            "statically registered test plugin"
        }

        fn configure(&mut self, config: HashMap<String, serde_json::Value>) -> Result<()> {
            self.config = config;
            Ok(())
        }

        fn get_config(&self) -> &HashMap<String, serde_json::Value> {
            &self.config
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn forward(&self, input: crate::tensor::Tensor) -> Result<crate::tensor::Tensor> {
            Ok(input)
        }
    }

    fn make_static_plugin() -> Result<Box<dyn Plugin>> {
        Ok(Box::new(StaticTestPlugin::default()))
    }

    /// Regression test: `LibraryHandle::load` returned a handle without opening
    /// anything and cached it, and `create_plugin` reported an error saying
    /// "not implemented in this example".
    #[test]
    fn test_dynamic_loading_is_refused_and_nothing_is_cached() {
        let loader = PluginLoader::new();
        let info = PluginInfo::new("ghost", "0.1.0", "never loaded", &[]);

        let error = loader.load_plugin(&info).expect_err("no dynamic loader is linked");
        let message = error.to_string();
        assert!(
            message.contains("register_static_plugin"),
            "the error must point at the supported path: {message}"
        );
        assert!(
            !message.contains("in this example"),
            "user-facing errors must not mention an example: {message}"
        );

        // Nothing was cached for a library that was never opened.
        let cache = loader.library_cache.lock().expect("lock");
        assert!(
            cache.is_empty(),
            "a failed load must not populate the cache"
        );
    }

    /// Regression test: `load_embedded_metadata` invented
    /// `PluginInfo::new(filename, "1.0.0", "Dynamically loaded plugin", &[])`.
    #[test]
    fn test_embedded_metadata_is_refused() {
        let loader = PluginLoader::new();
        let path =
            std::env::temp_dir().join(format!("trustformers_plugin_{}.so", std::process::id()));
        std::fs::write(&path, b"not a real plugin").expect("write failed");

        let error = loader
            .load_plugin_info(&path)
            .expect_err("no metadata file exists and none can be read from the binary");
        assert!(
            error.to_string().contains(".json"),
            "the error must say what is missing: {error}"
        );

        std::fs::remove_file(&path).ok();
    }

    /// A companion metadata file is still read.
    #[test]
    fn test_companion_metadata_is_read() {
        let loader = PluginLoader::new();
        let base =
            std::env::temp_dir().join(format!("trustformers_plugin_meta_{}", std::process::id()));
        let library = base.with_extension("so");
        let metadata = base.with_extension("json");

        std::fs::write(&library, b"binary").expect("write failed");
        std::fs::write(
            &metadata,
            serde_json::to_string(&PluginInfo::new("real", "2.1.0", "declared", &[]))
                .expect("serialize failed"),
        )
        .expect("write failed");

        let info = loader.load_plugin_info(&library).expect("metadata file must be read");
        assert_eq!(info.name(), "real");
        assert_eq!(
            info.version().to_string(),
            "2.1.0",
            "the version must come from the file, not a hardcoded 1.0.0"
        );

        std::fs::remove_file(&library).ok();
        std::fs::remove_file(&metadata).ok();
    }

    /// A statically registered plugin still loads.
    #[test]
    fn test_static_plugins_still_load() {
        let loader = PluginLoader::new();
        loader
            .register_static_plugin("static-one", make_static_plugin)
            .expect("registration failed");

        let info = PluginInfo::new("static-one", "1.0.0", "static", &[]);
        assert!(
            loader.load_plugin(&info).is_ok(),
            "static registration is the supported path and must work"
        );
    }
}
