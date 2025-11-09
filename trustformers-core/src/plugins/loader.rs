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

    /// Loads embedded metadata from a plugin file.
    fn load_embedded_metadata<P: AsRef<Path>>(&self, path: P) -> Result<PluginInfo> {
        // This is a simplified implementation
        // In a real implementation, you would read metadata from the plugin file
        // For now, we'll create basic info from the filename
        let path = path.as_ref();
        let name = path.file_stem().and_then(|s| s.to_str()).ok_or_else(|| {
            TrustformersError::plugin_error("Invalid plugin filename".to_string())
        })?;

        Ok(PluginInfo::new(
            name,
            "1.0.0",
            "Dynamically loaded plugin",
            &[],
        ))
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
    fn load(info: &PluginInfo) -> Result<Self> {
        // This is a simplified implementation
        // In a real implementation, you would use libloading or similar
        // to actually load the dynamic library

        Ok(Self {
            name: info.name().to_string(),
            _entry_point: info.entry_point().to_string(),
        })
    }

    /// Creates a plugin instance from this library.
    ///
    /// # Returns
    ///
    /// A boxed plugin instance.
    fn create_plugin(&self) -> Result<Box<dyn Plugin>> {
        // This is a simplified implementation
        // In a real implementation, you would resolve the plugin factory symbol
        // and call it to create the plugin instance

        Err(TrustformersError::plugin_error(
            "Dynamic plugin loading not implemented in this example".to_string(),
        ))
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
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// struct MyPlugin;
/// impl Plugin for MyPlugin {
///     // ... implementation
/// }
///
/// register_static_plugin!(MyPlugin, "my_plugin");
/// # Ok(())
/// # }
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
