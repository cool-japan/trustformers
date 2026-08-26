# React Native usage example (not a package)

This directory holds a **standalone TypeScript example**, not an npm package.
There is no `package.json`, no native bridge module source and no build script
here, so nothing in this directory is installable or publishable as-is.

`TrustformersCompleteExample.tsx` imports from `@trustformers/react-native`.
**That module is not published and its source is not in this repository.** The
import shows the API shape the Rust-side bridge (`src/react_native.rs`,
`src/react_native_turbo.rs`, `src/react_native_fabric.rs`, Cargo features
`react-native` / `expo`) is designed to expose; it is a specification of intent,
not a dependency you can currently resolve.

To run the example you would first have to write the JavaScript/TypeScript
package that wraps the Rust `cdylib` through a Turbo Module / JSI bridge and
publish it under that name.
