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
            let Some(target) = event.target() else { return };
            let Ok(request) = target.dyn_into::<IdbRequest>() else {
                return;
            };
            let Ok(result) = request.result() else { return };
            let _ = resolve.call1(&JsValue::NULL, &result);
        }) as Box<dyn FnMut(_)>);

        let error_callback = Closure::wrap(Box::new(move |event: web_sys::Event| {
            let Some(target) = event.target() else { return };
            let Ok(request) = target.dyn_into::<IdbRequest>() else {
                return;
            };
            let Ok(Some(error)) = request.error() else {
                return;
            };
            let _ = reject.call1(&JsValue::NULL, &error);
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
            let Some(target) = event.target() else { return };
            let Ok(request) = target.dyn_into::<IdbOpenDbRequest>() else {
                return;
            };
            let Ok(result) = request.result() else { return };
            let _ = resolve.call1(&JsValue::NULL, &result);
        }) as Box<dyn FnMut(_)>);

        let error_callback = Closure::wrap(Box::new(move |event: web_sys::Event| {
            let Some(target) = event.target() else { return };
            let Ok(request) = target.dyn_into::<IdbOpenDbRequest>() else {
                return;
            };
            let Ok(Some(error)) = request.error() else {
                return;
            };
            let _ = reject.call1(&JsValue::NULL, &error);
        }) as Box<dyn FnMut(_)>);

        request.set_onsuccess(Some(success_callback.as_ref().unchecked_ref()));
        request.set_onerror(Some(error_callback.as_ref().unchecked_ref()));

        success_callback.forget();
        error_callback.forget();
    })
}

/// Compression type recorded against a stored model's bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionType {
    /// Bytes stored exactly as given, uncompressed.
    None,
    /// **Legacy marker only** — no code in this crate ever writes this
    /// variant. Records tagged `Gzip` were written by an older version of
    /// [`ModelStorage::store_model`] that claimed gzip compression for any
    /// payload over 1MiB while actually storing the raw bytes unchanged
    /// (a bug, not a lossless-but-differently-named format). Reading a
    /// `Gzip` record must therefore return its bytes as-is rather than
    /// attempting to decompress them — see the legacy-honest-read branch
    /// in [`ModelStorage::get_model`]. New writes always use [`Self::Deflate`]
    /// instead.
    Gzip,
    /// Real DEFLATE (RFC 1951) compression via `oxiarc_deflate`, applied
    /// by [`ModelStorage::store_model`] to payloads over 1MiB. Every
    /// record tagged `Deflate` genuinely holds deflate-compressed bytes.
    Deflate,
    /// Declared for API completeness but not implemented: no code path in
    /// this crate ever writes a `Brotli` record, and reading one returns a
    /// structured error rather than silently returning undecoded bytes.
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
            compression_type: CompressionType::Deflate,
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
            let Some(target) = event.target() else { return };
            let Ok(request) = target.dyn_into::<IdbOpenDbRequest>() else {
                return;
            };
            let Ok(result) = request.result() else { return };
            let Ok(db) = result.dyn_into::<IdbDatabase>() else {
                return;
            };

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
                if let Ok(model_store) = db.create_object_store("models") {
                    // Create indices - use Reflect to call createIndex
                    let create_index_fn =
                        js_sys::Reflect::get(&model_store, &JsValue::from_str("createIndex"))
                            .unwrap_or(JsValue::UNDEFINED);
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
                let _ = db.create_object_store("metadata");
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

        // Compress data if it's large enough. Real DEFLATE via
        // `oxiarc_deflate` (already used the same way in
        // `runtime::edge_caching` and `storage::model_splitting`) — a
        // previous version stored the raw bytes here while labeling the
        // record `Gzip`, which is why that tag is now legacy-only (see
        // `CompressionType::Gzip`'s doc comment).
        let (compressed_data, compression_type) = match Self::compress_for_storage(data) {
            Ok(pair) => pair,
            Err(e) => {
                // Compression genuinely failing on arbitrary bytes is not
                // expected, but store the model uncompressed (labeled
                // honestly) rather than losing it.
                #[cfg(target_arch = "wasm32")]
                web_sys::console::log_1(
                    &format!(
                        "compression failed for model '{model_name}', storing uncompressed: {e}"
                    )
                    .into(),
                );
                #[cfg(not(target_arch = "wasm32"))]
                let _ = &e;
                (data.to_vec(), CompressionType::None)
            },
        };

        let now = Date::now();
        let checksum = Self::calculate_checksum(&compressed_data);

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

        // Verify checksum against the stored (possibly compressed) bytes,
        // before any decompression — see `Self::verify_checksum` for why
        // the algorithm depends on the checksum's own format.
        if !Self::verify_checksum(&stored_model.data, &stored_model.metadata.checksum) {
            return Err(format!(
                "Model data corruption detected for '{model_id}': checksum mismatch"
            )
            .into());
        }

        // Decompress data if needed (real core in `Self::decompress_stored_data`).
        let data = Self::decompress_stored_data(
            stored_model.data,
            &stored_model.metadata.compression_type,
            model_id,
        )
        .map_err(|e| JsValue::from_str(&e))?;

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
    //
    // `compress_for_storage`, `decompress_stored_data`, `calculate_checksum`,
    // `calculate_legacy_checksum` and `verify_checksum` are deliberately
    // plain functions over owned/borrowed bytes (no `js_sys`/`JsValue`,
    // no `&self`) so they can be exercised directly by native unit tests —
    // `store_model`/`get_model` themselves need a real `IdbDatabase` and so
    // cannot run outside a browser.

    /// Choose real compression for a payload about to be stored: DEFLATE
    /// (via `oxiarc_deflate`) for anything over 1MiB, stored raw otherwise.
    /// `Err` only if `oxiarc_deflate` itself fails (not expected for
    /// arbitrary bytes, but its API is fallible) — `store_model` treats
    /// that as "fall back to storing uncompressed", never as data loss.
    fn compress_for_storage(data: &[u8]) -> Result<(Vec<u8>, CompressionType), String> {
        if data.len() > 1024 * 1024 {
            let compressed = oxiarc_deflate::deflate(data, 6).map_err(|e| e.to_string())?;
            Ok((compressed, CompressionType::Deflate))
        } else {
            Ok((data.to_vec(), CompressionType::None))
        }
    }

    /// Inverse of [`Self::compress_for_storage`], dispatching on the
    /// record's stored `compression_type`. `model_id` is only used to
    /// build a readable error message.
    fn decompress_stored_data(
        data: Vec<u8>,
        compression_type: &CompressionType,
        model_id: &str,
    ) -> Result<Vec<u8>, String> {
        match compression_type {
            CompressionType::None => Ok(data),
            CompressionType::Deflate => oxiarc_deflate::inflate(&data)
                .map_err(|e| format!("failed to decompress model '{model_id}': {e}")),
            CompressionType::Gzip => {
                // Legacy marker only: records written before real
                // compression existed used this tag while the payload was
                // stored completely raw. Return the bytes as-is — do NOT
                // attempt to decompress them, see the doc comment on
                // `CompressionType::Gzip`.
                Ok(data)
            },
            CompressionType::Brotli => Err(format!(
                "model '{model_id}' is stored as Brotli, which this build cannot decode \
                 (no Brotli decompressor is implemented)"
            )),
        }
    }

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
        models.sort_by(|a, b| {
            a.last_accessed
                .partial_cmp(&b.last_accessed)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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

    /// Real SHA-256 checksum (same approach as
    /// `plugin_framework::calculate_plugin_checksum`) — 64 lowercase hex
    /// characters. Every checksum this method writes going forward uses
    /// this format; see [`Self::verify_checksum`] for reading records
    /// written by the older 8-hex-character byte-sum checksum.
    fn calculate_checksum(data: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Legacy checksum: a plain sum of byte values, formatted as 8 hex
    /// characters — what every record wrote before this file used SHA-256.
    /// Weak (order-insensitive, no cryptographic guarantee) but genuinely
    /// computed; kept only so records written before this change can still
    /// be read and their integrity checked, not to produce new checksums.
    fn calculate_legacy_checksum(data: &[u8]) -> String {
        let sum: u32 = data.iter().map(|&b| b as u32).sum();
        format!("{sum:08x}")
    }

    /// Verify `data` against a stored checksum, dispatching on the
    /// checksum's own format: 64 hex characters means real SHA-256 (every
    /// record this file writes today); anything shorter (8 hex characters
    /// in every record actually produced by the legacy code) is checked
    /// against the legacy byte-sum instead, so pre-existing IndexedDB
    /// records remain readable rather than being rejected as "corrupt"
    /// purely because the checksum algorithm changed.
    fn verify_checksum(data: &[u8], stored_checksum: &str) -> bool {
        if stored_checksum.len() == 64 {
            Self::calculate_checksum(data) == stored_checksum
        } else {
            Self::calculate_legacy_checksum(data) == stored_checksum
        }
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

    // -----------------------------------------------------------------
    // Real compression (replacing the old "store raw, label it Gzip" bug).
    // -----------------------------------------------------------------

    #[test]
    fn test_compress_for_storage_stores_small_payloads_raw_as_none() {
        let data = std::vec![1u8, 2, 3, 4];
        let (stored, kind) = ModelStorage::compress_for_storage(&data).unwrap();
        assert_eq!(stored, data);
        assert!(matches!(kind, CompressionType::None));
    }

    #[test]
    fn test_compress_for_storage_deflates_large_payloads_for_real() {
        // Highly repetitive so DEFLATE engages hard — a regression guard
        // against the old bug where "compression" was a no-op that
        // returned the input unchanged while still labeling it `Gzip`.
        let data = std::vec![0x11u8; 2 * 1024 * 1024];
        let (stored, kind) = ModelStorage::compress_for_storage(&data).unwrap();
        assert!(matches!(kind, CompressionType::Deflate));
        assert!(
            stored.len() < data.len() / 10,
            "highly repetitive >1MiB data must compress substantially: {} of {} bytes",
            stored.len(),
            data.len()
        );
        assert_ne!(
            stored, data,
            "a real compressor must not just return the input unchanged"
        );
    }

    #[test]
    fn test_compress_then_decompress_round_trips_exactly() {
        let mut state = 321u32;
        let data: Vec<u8> = (0..1_500_000)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                (state >> 16) as u8
            })
            .collect();
        let (stored, kind) = ModelStorage::compress_for_storage(&data).unwrap();
        assert!(matches!(kind, CompressionType::Deflate));
        let recovered = ModelStorage::decompress_stored_data(stored, &kind, "m1")
            .expect("a real Deflate record must decompress cleanly");
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_decompress_none_returns_bytes_unchanged() {
        let data = std::vec![9u8, 8, 7];
        let out = ModelStorage::decompress_stored_data(data.clone(), &CompressionType::None, "m1")
            .unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_decompress_legacy_gzip_tag_returns_raw_bytes_not_a_decode_attempt() {
        // The whole point of keeping `Gzip` as a legacy marker: a record
        // tagged `Gzip` never actually held gzip- or deflate-compressed
        // bytes (that was the bug), so decoding it must be a no-op, not an
        // attempt to run a decompressor over data that was never
        // compressed (which would corrupt or error on real payloads).
        let raw_uncompressed_payload = std::vec![0xABu8; 4096];
        let out = ModelStorage::decompress_stored_data(
            raw_uncompressed_payload.clone(),
            &CompressionType::Gzip,
            "legacy-model",
        )
        .expect("legacy Gzip-tagged records must read back cleanly, not error");
        assert_eq!(out, raw_uncompressed_payload);
    }

    #[test]
    fn test_decompress_brotli_is_a_structured_error_not_silent_passthrough() {
        let data = std::vec![1u8, 2, 3];
        let err = ModelStorage::decompress_stored_data(data, &CompressionType::Brotli, "m1")
            .expect_err("Brotli must error, not silently return undecoded bytes");
        assert!(err.contains("Brotli"));
        assert!(err.contains("m1"));
    }

    // -----------------------------------------------------------------
    // Real SHA-256 checksums, with legacy-format compatibility.
    // -----------------------------------------------------------------

    #[test]
    fn test_calculate_checksum_is_64_hex_chars() {
        let checksum = ModelStorage::calculate_checksum(b"hello model bytes");
        assert_eq!(checksum.len(), 64);
        assert!(checksum.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_calculate_checksum_matches_known_sha256_vector() {
        // SHA-256("abc") is a well-known test vector; confirms this is
        // real SHA-256, not the old byte-sum reformatted to look similar.
        assert_eq!(
            ModelStorage::calculate_checksum(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn test_verify_checksum_accepts_real_sha256_and_rejects_corruption() {
        let data = std::vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let checksum = ModelStorage::calculate_checksum(&data);
        assert!(ModelStorage::verify_checksum(&data, &checksum));

        let mut corrupted = data.clone();
        corrupted[0] ^= 0xFF;
        assert!(!ModelStorage::verify_checksum(&corrupted, &checksum));
    }

    #[test]
    fn test_verify_checksum_accepts_legacy_byte_sum_format() {
        // Simulates a record written by the pre-SHA-256 code: an 8-hex-char
        // checksum. It must still verify against the legacy algorithm so
        // old IndexedDB records are not rejected as "corrupt" just because
        // the checksum algorithm changed going forward.
        let data = std::vec![10u8, 20, 30];
        let legacy_checksum = ModelStorage::calculate_legacy_checksum(&data);
        assert_eq!(legacy_checksum.len(), 8);
        assert!(ModelStorage::verify_checksum(&data, &legacy_checksum));

        let mut corrupted = data.clone();
        corrupted[0] ^= 0xFF;
        assert!(!ModelStorage::verify_checksum(&corrupted, &legacy_checksum));
    }
}
