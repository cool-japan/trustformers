//! IndexedDB model storage for browser caching and offline usage

use js_sys::{Array, Date};
use serde::{Deserialize, Serialize};
use std::boxed::Box;
use std::format;
use std::string::{String, ToString};
use std::vec::Vec;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{IdbDatabase, IdbOpenDbRequest, IdbRequest, IdbTransaction, IdbVersionChangeEvent};

use super::StorageError;

/// Helper function to convert IdbRequest to a Promise
fn request_to_promise(request: &IdbRequest) -> js_sys::Promise {
    js_sys::Promise::new(&mut |resolve, reject| {
        let success_callback = Closure::wrap(Box::new(move |event: web_sys::Event| {
            let target = event.target().unwrap();
            let request: IdbRequest = target.dyn_into().unwrap();
            let result = request.result().unwrap();
            resolve.call1(&JsValue::NULL, &result).unwrap();
        }) as Box<dyn FnMut(_)>);

        let error_callback = Closure::wrap(Box::new(move |event: web_sys::Event| {
            let target = event.target().unwrap();
            let request: IdbRequest = target.dyn_into().unwrap();
            let error = request.error().unwrap().unwrap();
            reject.call1(&JsValue::NULL, &error).unwrap();
        }) as Box<dyn FnMut(_)>);

        request.set_onsuccess(Some(success_callback.as_ref().unchecked_ref()));
        request.set_onerror(Some(error_callback.as_ref().unchecked_ref()));

        success_callback.forget();
        error_callback.forget();
    })
}

/// Helper function to convert IdbOpenDbRequest to a Promise
fn open_request_to_promise(request: &IdbOpenDbRequest) -> js_sys::Promise {
    js_sys::Promise::new(&mut |resolve, reject| {
        let success_callback = Closure::wrap(Box::new(move |event: web_sys::Event| {
            let target = event.target().unwrap();
            let request: IdbOpenDbRequest = target.dyn_into().unwrap();
            let result = request.result().unwrap();
            resolve.call1(&JsValue::NULL, &result).unwrap();
        }) as Box<dyn FnMut(_)>);

        let error_callback = Closure::wrap(Box::new(move |event: web_sys::Event| {
            let target = event.target().unwrap();
            let request: IdbOpenDbRequest = target.dyn_into().unwrap();
            let error = request.error().unwrap().unwrap();
            reject.call1(&JsValue::NULL, &error).unwrap();
        }) as Box<dyn FnMut(_)>);

        request.set_onsuccess(Some(success_callback.as_ref().unchecked_ref()));
        request.set_onerror(Some(error_callback.as_ref().unchecked_ref()));

        success_callback.forget();
        error_callback.forget();
    })
}

/// Compression type for model storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionType {
    None,
    Gzip,
    Brotli,
}

/// Storage configuration for IndexedDB
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub db_name: String,
    pub max_storage_mb: f64,
    pub enable_compression: bool,
    pub compression_type: CompressionType,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            db_name: "trustformers_models".to_string(),
            max_storage_mb: 1024.0,
            enable_compression: true,
            compression_type: CompressionType::Gzip,
        }
    }
}

/// Initialize the IndexedDB module
pub fn initialize() -> Result<(), StorageError> {
    // Check if IndexedDB is available
    if let Some(window) = web_sys::window() {
        if window.indexed_db().is_err() {
            return Err(StorageError::InitializationError(
                "IndexedDB is not supported in this browser".to_string(),
            ));
        }
    } else {
        return Err(StorageError::InitializationError(
            "No window object available".to_string(),
        ));
    }
    Ok(())
}

/// Model metadata for storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: String,
    pub name: String,
    pub version: String,
    pub architecture: String,
    pub size_bytes: usize,
    pub created_at: f64,
    pub last_accessed: f64,
    pub compression_type: CompressionType,
    pub checksum: String,
}

/// Stored model data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredModel {
    pub metadata: ModelMetadata,
    pub data: Vec<u8>,
}

/// IndexedDB-based model storage
#[wasm_bindgen]
pub struct ModelStorage {
    db_name: String,
    db_version: u32,
    max_storage_mb: f64,
    db: Option<IdbDatabase>,
}

#[wasm_bindgen]
impl ModelStorage {
    /// Create a new model storage instance
    #[wasm_bindgen(constructor)]
    pub fn new(db_name: String, max_storage_mb: f64) -> Self {
        Self {
            db_name,
            db_version: 1,
            max_storage_mb,
            db: None,
        }
    }

    /// Initialize the IndexedDB database
    pub async fn initialize(&mut self) -> Result<(), JsValue> {
        let window = web_sys::window().ok_or("No global window exists")?;
        let idb_factory = window
            .indexed_db()
            .map_err(|_| "IndexedDB not supported")?
            .ok_or("IndexedDB not available")?;

        let db_request = idb_factory
            .open_with_u32(&self.db_name, self.db_version)
            .map_err(|e| format!("Failed to open database: {:?}", e))?;

        // Set up database upgrade handler
        let upgrade_callback = Closure::wrap(Box::new(move |event: IdbVersionChangeEvent| {
            let target = event.target().unwrap();
            let request: IdbOpenDbRequest = target.dyn_into().unwrap();
            let db = request.result().unwrap().dyn_into::<IdbDatabase>().unwrap();

            // Create object stores
            // Use Reflect to access objectStoreNames property
            let store_names_obj = js_sys::Reflect::get(&db, &JsValue::from_str("objectStoreNames"))
                .unwrap_or(JsValue::NULL);
            let has_models = if !store_names_obj.is_null() && !store_names_obj.is_undefined() {
                let contains_fn =
                    js_sys::Reflect::get(&store_names_obj, &JsValue::from_str("contains"))
                        .ok()
                        .and_then(|f| f.dyn_into::<js_sys::Function>().ok());
                if let Some(contains_fn) = contains_fn {
                    contains_fn
                        .call1(&store_names_obj, &JsValue::from_str("models"))
                        .ok()
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false)
                } else {
                    false
                }
            } else {
                false
            };

            if !has_models {
                // IdbObjectStoreParameters not available in web-sys 0.3.81 - using default
                let model_store = db.create_object_store("models").unwrap();

                // Create indices - use Reflect to call createIndex
                let create_index_fn =
                    js_sys::Reflect::get(&model_store, &JsValue::from_str("createIndex")).unwrap();
                let create_index_fn: &js_sys::Function = create_index_fn.unchecked_ref();
                let _ = create_index_fn.call2(
                    &model_store,
                    &JsValue::from_str("name"),
                    &JsValue::from_str("name"),
                );
                let _ = create_index_fn.call2(
                    &model_store,
                    &JsValue::from_str("last_accessed"),
                    &JsValue::from_str("last_accessed"),
                );
                let _ = create_index_fn.call2(
                    &model_store,
                    &JsValue::from_str("size_bytes"),
                    &JsValue::from_str("size_bytes"),
                );
            }

            let has_metadata = if !store_names_obj.is_null() && !store_names_obj.is_undefined() {
                let contains_fn =
                    js_sys::Reflect::get(&store_names_obj, &JsValue::from_str("contains"))
                        .ok()
                        .and_then(|f| f.dyn_into::<js_sys::Function>().ok());
                if let Some(contains_fn) = contains_fn {
                    contains_fn
                        .call1(&store_names_obj, &JsValue::from_str("metadata"))
                        .ok()
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false)
                } else {
                    false
                }
            } else {
                false
            };

            if !has_metadata {
                // IdbObjectStoreParameters not available in web-sys 0.3.81 - using default
                db.create_object_store("metadata").unwrap();
            }
        }) as Box<dyn FnMut(_)>);

        db_request.set_onupgradeneeded(Some(upgrade_callback.as_ref().unchecked_ref()));
        upgrade_callback.forget();

        let db_promise = open_request_to_promise(&db_request);
        let db_result = JsFuture::from(db_promise).await?;
        let db: IdbDatabase = db_result.dyn_into()?;

        self.db = Some(db);
        Ok(())
    }

    /// Store a model in IndexedDB
    pub async fn store_model(
        &self,
        model_id: &str,
        model_name: &str,
        architecture: &str,
        version: &str,
        data: &[u8],
    ) -> Result<(), JsValue> {
        let db = self.db.as_ref().ok_or("Database not initialized")?;

        // Check storage space before storing
        self.ensure_storage_space(data.len()).await?;

        // Compress data if it's large enough
        let (compressed_data, compression_type) = if data.len() > 1024 * 1024 {
            // For models > 1MB, apply compression
            // In a real implementation, you'd use actual compression
            (data.to_vec(), CompressionType::Gzip)
        } else {
            (data.to_vec(), CompressionType::None)
        };

        let now = Date::now();
        let checksum = self.calculate_checksum(&compressed_data);

        let metadata = ModelMetadata {
            id: model_id.to_string(),
            name: model_name.to_string(),
            version: version.to_string(),
            architecture: architecture.to_string(),
            size_bytes: compressed_data.len(),
            created_at: now,
            last_accessed: now,
            compression_type,
            checksum,
        };

        let stored_model = StoredModel {
            metadata,
            data: compressed_data,
        };

        // Convert to JS object for storage
        let js_object = serde_wasm_bindgen::to_value(&stored_model)?;

        // Use Reflect to call transaction() method with store names array and mode
        let transaction_fn = js_sys::Reflect::get(db, &JsValue::from_str("transaction"))?;
        let transaction_fn: &js_sys::Function = transaction_fn.unchecked_ref();
        let store_names = js_sys::Array::of1(&"models".into());
        let transaction_result =
            transaction_fn.call2(db, &store_names, &JsValue::from_str("readwrite"))?;
        let transaction: IdbTransaction = transaction_result.dyn_into()?;
        let object_store = transaction.object_store("models")?;

        let request = object_store.put(&js_object)?;
        let request_promise = request_to_promise(&request);
        let _result = JsFuture::from(request_promise).await?;

        web_sys::console::log_1(
            &format!(
                "Stored model '{}' ({} bytes, {:?} compression)",
                model_name,
                stored_model.data.len(),
                stored_model.metadata.compression_type
            )
            .into(),
        );

        Ok(())
    }

    /// Retrieve a model from IndexedDB
    pub async fn get_model(&self, model_id: &str) -> Result<Option<Vec<u8>>, JsValue> {
        let db = self.db.as_ref().ok_or("Database not initialized")?;

        let transaction = db.transaction_with_str("models")?;
        let object_store = transaction.object_store("models")?;

        let request = object_store.get(&model_id.into())?;
        let request_promise = request_to_promise(&request);
        let result = JsFuture::from(request_promise).await?;

        if result.is_undefined() {
            return Ok(None);
        }

        let stored_model: StoredModel = serde_wasm_bindgen::from_value(result)?;

        // Update last accessed time
        self.update_last_accessed(model_id).await?;

        // Verify checksum
        let calculated_checksum = self.calculate_checksum(&stored_model.data);
        if calculated_checksum != stored_model.metadata.checksum {
            return Err("Model data corruption detected".into());
        }

        // Decompress data if needed
        let data = match stored_model.metadata.compression_type {
            CompressionType::None => stored_model.data,
            CompressionType::Gzip => {
                // In a real implementation, you'd decompress here
                stored_model.data
            },
            CompressionType::Brotli => {
                // In a real implementation, you'd decompress here
                stored_model.data
            },
        };

        web_sys::console::log_1(
            &format!(
                "Retrieved model '{}' ({} bytes)",
                stored_model.metadata.name,
                data.len()
            )
            .into(),
        );

        Ok(Some(data))
    }

    /// List all stored models
    pub async fn list_models(&self) -> Result<JsValue, JsValue> {
        let db = self.db.as_ref().ok_or("Database not initialized")?;

        let transaction = db.transaction_with_str("models")?;
        let object_store = transaction.object_store("models")?;

        let request = object_store.get_all()?;
        let request_promise = request_to_promise(&request);
        let result = JsFuture::from(request_promise).await?;

        let js_array: Array = result.dyn_into()?;
        let mut models = Vec::new();

        for i in 0..js_array.length() {
            let item = js_array.get(i);
            let stored_model: StoredModel = serde_wasm_bindgen::from_value(item)?;
            models.push(stored_model.metadata);
        }

        // Convert Vec to JsValue using serde
        serde_wasm_bindgen::to_value(&models).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Delete a model from storage
    pub async fn delete_model(&self, model_id: &str) -> Result<(), JsValue> {
        let db = self.db.as_ref().ok_or("Database not initialized")?;

        // Use Reflect to call transaction() method with store names array and mode
        let transaction_fn = js_sys::Reflect::get(db, &JsValue::from_str("transaction"))?;
        let transaction_fn: &js_sys::Function = transaction_fn.unchecked_ref();
        let store_names = js_sys::Array::of1(&"models".into());
        let transaction_result =
            transaction_fn.call2(db, &store_names, &JsValue::from_str("readwrite"))?;
        let transaction: IdbTransaction = transaction_result.dyn_into()?;
        let object_store = transaction.object_store("models")?;

        let request = object_store.delete(&model_id.into())?;
        let request_promise = request_to_promise(&request);
        let _result = JsFuture::from(request_promise).await?;

        web_sys::console::log_1(&format!("Deleted model '{}'", model_id).into());

        Ok(())
    }

    /// Get total storage usage in bytes
    pub async fn get_storage_usage(&self) -> Result<usize, JsValue> {
        let models_js = self.list_models().await?;
        let models: Vec<ModelMetadata> = serde_wasm_bindgen::from_value(models_js)?;
        let total_size = models.iter().map(|m| m.size_bytes).sum();
        Ok(total_size)
    }

    /// Clear all stored models
    pub async fn clear_all(&self) -> Result<(), JsValue> {
        let db = self.db.as_ref().ok_or("Database not initialized")?;

        // Use Reflect to call transaction() method with store names array and mode
        let transaction_fn = js_sys::Reflect::get(db, &JsValue::from_str("transaction"))?;
        let transaction_fn: &js_sys::Function = transaction_fn.unchecked_ref();
        let store_names = js_sys::Array::of1(&"models".into());
        let transaction_result =
            transaction_fn.call2(db, &store_names, &JsValue::from_str("readwrite"))?;
        let transaction: IdbTransaction = transaction_result.dyn_into()?;
        let object_store = transaction.object_store("models")?;

        let request = object_store.clear()?;
        let request_promise = request_to_promise(&request);
        let _result = JsFuture::from(request_promise).await?;

        web_sys::console::log_1(&"Cleared all stored models".into());

        Ok(())
    }

    /// Check if a model exists in storage
    pub async fn has_model(&self, model_id: &str) -> Result<bool, JsValue> {
        let db = self.db.as_ref().ok_or("Database not initialized")?;

        let transaction = db.transaction_with_str("models")?;
        let object_store = transaction.object_store("models")?;

        let request = object_store.count_with_key(&model_id.into())?;
        let request_promise = request_to_promise(&request);
        let result = JsFuture::from(request_promise).await?;

        let count: f64 = result.as_f64().unwrap_or(0.0);
        Ok(count > 0.0)
    }

    // Private helper methods

    async fn ensure_storage_space(&self, required_bytes: usize) -> Result<(), JsValue> {
        let current_usage = self.get_storage_usage().await?;
        let max_bytes = (self.max_storage_mb * 1024.0 * 1024.0) as usize;

        if current_usage + required_bytes > max_bytes {
            // Implement LRU eviction
            self.evict_lru_models(required_bytes).await?;
        }

        Ok(())
    }

    async fn evict_lru_models(&self, required_bytes: usize) -> Result<(), JsValue> {
        let models_js = self.list_models().await?;
        let mut models: Vec<ModelMetadata> = serde_wasm_bindgen::from_value(models_js)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Sort by last accessed time (oldest first)
        models.sort_by(|a, b| a.last_accessed.partial_cmp(&b.last_accessed).unwrap());

        let mut freed_bytes = 0;
        for model in models {
            if freed_bytes >= required_bytes {
                break;
            }

            self.delete_model(&model.id).await?;
            freed_bytes += model.size_bytes;

            web_sys::console::log_1(
                &format!("Evicted model '{}' to free space", model.name).into(),
            );
        }

        Ok(())
    }

    async fn update_last_accessed(&self, model_id: &str) -> Result<(), JsValue> {
        let db = self.db.as_ref().ok_or("Database not initialized")?;

        // Use Reflect to call transaction() method with store names array and mode
        let transaction_fn = js_sys::Reflect::get(db, &JsValue::from_str("transaction"))?;
        let transaction_fn: &js_sys::Function = transaction_fn.unchecked_ref();
        let store_names = js_sys::Array::of1(&"models".into());
        let transaction_result =
            transaction_fn.call2(db, &store_names, &JsValue::from_str("readwrite"))?;
        let transaction: IdbTransaction = transaction_result.dyn_into()?;
        let object_store = transaction.object_store("models")?;

        let get_request = object_store.get(&model_id.into())?;
        let get_promise = request_to_promise(&get_request);
        let result = JsFuture::from(get_promise).await?;

        if !result.is_undefined() {
            let mut stored_model: StoredModel = serde_wasm_bindgen::from_value(result)?;
            stored_model.metadata.last_accessed = Date::now();

            let js_object = serde_wasm_bindgen::to_value(&stored_model)?;
            let put_request = object_store.put(&js_object)?;
            let put_promise = request_to_promise(&put_request);
            let _result = JsFuture::from(put_promise).await?;
        }

        Ok(())
    }

    fn calculate_checksum(&self, data: &[u8]) -> String {
        // Simple checksum calculation
        // In a real implementation, you'd use a proper hash function like SHA-256
        let sum: u32 = data.iter().map(|&b| b as u32).sum();
        format!("{:08x}", sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_creation() {
        let metadata = ModelMetadata {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            version: "1.0.0".to_string(),
            architecture: "BERT".to_string(),
            size_bytes: 1024,
            created_at: 0.0,
            last_accessed: 0.0,
            compression_type: CompressionType::None,
            checksum: "abcd1234".to_string(),
        };

        assert_eq!(metadata.id, "test-model");
        assert_eq!(metadata.size_bytes, 1024);
    }
}
