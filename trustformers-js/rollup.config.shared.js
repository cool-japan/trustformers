import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import terser from '@rollup/plugin-terser';
import peerDepsExternal from 'rollup-plugin-peer-deps-external';

export function createConfig(packageName, external = []) {
  const isProduction = process.env.NODE_ENV === 'production';
  
  return {
    input: 'src/index.js',
    output: [
      {
        file: 'dist/index.cjs.js',
        format: 'cjs',
        exports: 'named',
        sourcemap: true
      },
      {
        file: 'dist/index.esm.js',
        format: 'esm',
        sourcemap: true
      }
    ],
    external: [
      ...external,
      /@trustformers\/.*/,
      /node_modules/,
      '../pkg/trustformers_wasm.js' // Make WASM import external for development
    ],
    plugins: [
      peerDepsExternal(),
      resolve({
        browser: true,
        preferBuiltins: false
      }),
      commonjs(),
      isProduction && terser({
        compress: {
          drop_console: true,
          drop_debugger: true
        }
      })
    ].filter(Boolean),
    onwarn: (warning, warn) => {
      if (warning.code === 'THIS_IS_UNDEFINED') return;
      warn(warning);
    }
  };
}