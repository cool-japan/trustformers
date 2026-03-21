/**
 * WebGL Backend for TrustformeRS
 * Provides GPU acceleration for browsers without WebGPU support
 */

let wasmModule = null;

/**
 * Initialize WebGL backend
 * @param {Object} module - WASM module reference
 */
export function initWebGLBackend(module) {
  wasmModule = module;
}

/**
 * WebGL Backend for tensor operations
 */
export class WebGLBackend {
  constructor() {
    this.gl = null;
    this.programs = new Map();
    this.buffers = new Map();
    this.textures = new Map();
    this.framebuffers = new Map();
    this.supported = false;
    this.maxTextureSize = 0;
    this.extensions = {};
  }

  /**
   * Initialize WebGL context and check capabilities
   * @param {HTMLCanvasElement} canvas - Canvas element (optional)
   * @returns {Promise<boolean>} Success status
   */
  async initialize(canvas = null) {
    try {
      // Create or use provided canvas
      const canvasElement = canvas || document.createElement('canvas');
      canvasElement.width = 1;
      canvasElement.height = 1;
      
      // Try WebGL2 first, fallback to WebGL1
      this.gl = canvasElement.getContext('webgl2') || 
                canvasElement.getContext('webgl') ||
                canvasElement.getContext('experimental-webgl');
      
      if (!this.gl) {
        console.warn('WebGL not supported');
        return false;
      }

      // Check required extensions
      this.extensions = {
        floatTextures: this.gl.getExtension('OES_texture_float') || 
                      this.gl.getExtension('EXT_color_buffer_float'),
        textureFloat: this.gl.getExtension('OES_texture_float_linear'),
        vertexArrays: this.gl.getExtension('OES_vertex_array_object'),
        instancedArrays: this.gl.getExtension('ANGLE_instanced_arrays')
      };

      // Get capabilities
      this.maxTextureSize = this.gl.getParameter(this.gl.MAX_TEXTURE_SIZE);
      const maxFragmentTextures = this.gl.getParameter(this.gl.MAX_TEXTURE_IMAGE_UNITS);
      const maxVertexTextures = this.gl.getParameter(this.gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS);

      console.warn(`WebGL Backend initialized:`);
      console.warn(`- Context: ${this.gl.constructor.name}`);
      console.warn(`- Max texture size: ${this.maxTextureSize}`);
      console.warn(`- Max fragment textures: ${maxFragmentTextures}`);
      console.warn(`- Max vertex textures: ${maxVertexTextures}`);
      console.warn(`- Float textures: ${!!this.extensions.floatTextures}`);

      this.supported = true;
      
      // Initialize basic shaders
      await this.initializeShaders();
      
      return true;
    } catch (error) {
      console.error('Failed to initialize WebGL backend:', error);
      return false;
    }
  }

  /**
   * Initialize basic compute shaders
   */
  async initializeShaders() {
    // Vertex shader (common for all operations)
    const vertexShaderSource = `
      attribute vec2 a_position;
      attribute vec2 a_texCoord;
      varying vec2 v_texCoord;
      
      void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
        v_texCoord = a_texCoord;
      }
    `;

    // Basic matrix multiplication shader
    const matmulFragmentShader = `
      precision highp float;
      varying vec2 v_texCoord;
      uniform sampler2D u_matrixA;
      uniform sampler2D u_matrixB;
      uniform vec2 u_sizeA;
      uniform vec2 u_sizeB;
      uniform vec2 u_outputSize;
      
      void main() {
        vec2 pos = v_texCoord * u_outputSize;
        float row = floor(pos.y);
        float col = floor(pos.x);
        
        float sum = 0.0;
        for (float k = 0.0; k < u_sizeA.x; k += 1.0) {
          vec2 coordA = vec2(k / u_sizeA.x, row / u_sizeA.y);
          vec2 coordB = vec2(col / u_sizeB.x, k / u_sizeB.y);
          
          float a = texture2D(u_matrixA, coordA).r;
          float b = texture2D(u_matrixB, coordB).r;
          sum += a * b;
        }
        
        gl_FragColor = vec4(sum, 0.0, 0.0, 1.0);
      }
    `;

    // Element-wise operations shader
    const elementWiseFragmentShader = `
      precision highp float;
      varying vec2 v_texCoord;
      uniform sampler2D u_textureA;
      uniform sampler2D u_textureB;
      uniform int u_operation; // 0=add, 1=sub, 2=mul, 3=div
      
      void main() {
        vec4 a = texture2D(u_textureA, v_texCoord);
        vec4 b = texture2D(u_textureB, v_texCoord);
        
        vec4 result;
        if (u_operation == 0) {
          result = a + b;
        } else if (u_operation == 1) {
          result = a - b;
        } else if (u_operation == 2) {
          result = a * b;
        } else if (u_operation == 3) {
          result = a / b;
        } else {
          result = a;
        }
        
        gl_FragColor = result;
      }
    `;

    // Activation functions shader
    const activationFragmentShader = `
      precision highp float;
      varying vec2 v_texCoord;
      uniform sampler2D u_texture;
      uniform int u_activation; // 0=relu, 1=sigmoid, 2=tanh, 3=gelu
      
      void main() {
        vec4 x = texture2D(u_texture, v_texCoord);
        vec4 result;
        
        if (u_activation == 0) {
          // ReLU
          result = max(x, vec4(0.0));
        } else if (u_activation == 1) {
          // Sigmoid
          result = 1.0 / (1.0 + exp(-x));
        } else if (u_activation == 2) {
          // Tanh
          result = tanh(x);
        } else if (u_activation == 3) {
          // GELU approximation
          result = 0.5 * x * (1.0 + tanh(sqrt(2.0 / 3.14159) * (x + 0.044715 * x * x * x)));
        } else {
          result = x;
        }
        
        gl_FragColor = result;
      }
    `;

    // Create shader programs
    this.programs.set('matmul', this.createProgram(vertexShaderSource, matmulFragmentShader));
    this.programs.set('elementwise', this.createProgram(vertexShaderSource, elementWiseFragmentShader));
    this.programs.set('activation', this.createProgram(vertexShaderSource, activationFragmentShader));

    // Create quad buffer for rendering
    this.createQuadBuffer();
  }

  /**
   * Create a shader program
   * @param {string} vertexSource - Vertex shader source
   * @param {string} fragmentSource - Fragment shader source
   * @returns {WebGLProgram} Compiled program
   */
  createProgram(vertexSource, fragmentSource) {
    const vertexShader = this.createShader(this.gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = this.createShader(this.gl.FRAGMENT_SHADER, fragmentSource);
    
    const program = this.gl.createProgram();
    this.gl.attachShader(program, vertexShader);
    this.gl.attachShader(program, fragmentShader);
    this.gl.linkProgram(program);
    
    if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
      const error = this.gl.getProgramInfoLog(program);
      this.gl.deleteProgram(program);
      throw new Error(`Program linking failed: ${error}`);
    }
    
    return program;
  }

  /**
   * Create a shader
   * @param {number} type - Shader type
   * @param {string} source - Shader source
   * @returns {WebGLShader} Compiled shader
   */
  createShader(type, source) {
    const shader = this.gl.createShader(type);
    this.gl.shaderSource(shader, source);
    this.gl.compileShader(shader);
    
    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      const error = this.gl.getShaderInfoLog(shader);
      this.gl.deleteShader(shader);
      throw new Error(`Shader compilation failed: ${error}`);
    }
    
    return shader;
  }

  /**
   * Create quad buffer for full-screen rendering
   */
  createQuadBuffer() {
    const positions = new Float32Array([
      -1, -1,  0, 0,
       1, -1,  1, 0,
      -1,  1,  0, 1,
       1,  1,  1, 1
    ]);
    
    const buffer = this.gl.createBuffer();
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);
    
    this.buffers.set('quad', buffer);
  }

  /**
   * Create texture from tensor data
   * @param {Float32Array} data - Tensor data
   * @param {number} width - Texture width
   * @param {number} height - Texture height
   * @returns {WebGLTexture} Created texture
   */
  createTexture(data, width, height) {
    const texture = this.gl.createTexture();
    this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
    
    // Set texture parameters
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
    
    // Upload data
    const format = this.gl.RGBA;
    const type = this.gl.FLOAT;
    
    if (this.extensions.floatTextures) {
      this.gl.texImage2D(this.gl.TEXTURE_2D, 0, format, width, height, 0, format, type, data);
    } else {
      // Fallback to UNSIGNED_BYTE if float textures not supported
      const byteData = new Uint8Array(data.length * 4);
      for (let i = 0; i < data.length; i++) {
        const normalized = Math.max(0, Math.min(255, Math.floor(data[i] * 255)));
        byteData[i * 4] = normalized;
        byteData[(i * 4) + 1] = normalized;
        byteData[(i * 4) + 2] = normalized;
        byteData[i * 4 + 3] = 255;
      }
      this.gl.texImage2D(this.gl.TEXTURE_2D, 0, format, width, height, 0, format, this.gl.UNSIGNED_BYTE, byteData);
    }
    
    return texture;
  }

  /**
   * Perform matrix multiplication using WebGL
   * @param {Object} tensorA - First tensor
   * @param {Object} tensorB - Second tensor
   * @returns {Object} Result tensor
   */
  async matmul(tensorA, tensorB) {
    if (!this.supported) {
      throw new Error('WebGL backend not initialized');
    }

    const program = this.programs.get('matmul');
    this.gl.useProgram(program);

    // Get tensor data and shapes
    const dataA = await tensorA.to_js_array();
    const dataB = await tensorB.to_js_array();
    const shapeA = await tensorA.shape();
    const shapeB = await tensorB.shape();

    // Create textures
    const textureA = this.createTexture(new Float32Array(dataA), shapeA[1], shapeA[0]);
    const textureB = this.createTexture(new Float32Array(dataB), shapeB[1], shapeB[0]);

    // Set uniforms
    const locations = {
      matrixA: this.gl.getUniformLocation(program, 'u_matrixA'),
      matrixB: this.gl.getUniformLocation(program, 'u_matrixB'),
      sizeA: this.gl.getUniformLocation(program, 'u_sizeA'),
      sizeB: this.gl.getUniformLocation(program, 'u_sizeB'),
      outputSize: this.gl.getUniformLocation(program, 'u_outputSize')
    };

    this.gl.uniform1i(locations.matrixA, 0);
    this.gl.uniform1i(locations.matrixB, 1);
    this.gl.uniform2f(locations.sizeA, shapeA[1], shapeA[0]);
    this.gl.uniform2f(locations.sizeB, shapeB[1], shapeB[0]);
    this.gl.uniform2f(locations.outputSize, shapeB[1], shapeA[0]);

    // Bind textures
    this.gl.activeTexture(this.gl.TEXTURE0);
    this.gl.bindTexture(this.gl.TEXTURE_2D, textureA);
    this.gl.activeTexture(this.gl.TEXTURE1);
    this.gl.bindTexture(this.gl.TEXTURE_2D, textureB);

    // Setup framebuffer for output
    const [, outputWidth] = shapeB;
    const [outputHeight] = shapeA;
    const { framebuffer, texture: outputTexture } = this.createFramebuffer(outputWidth, outputHeight);

    // Render
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, framebuffer);
    this.gl.viewport(0, 0, outputWidth, outputHeight);
    this.renderQuad(program);

    // Read result
    const result = this.readFramebuffer(outputWidth, outputHeight);

    // Cleanup
    this.gl.deleteTexture(textureA);
    this.gl.deleteTexture(textureB);
    this.gl.deleteTexture(outputTexture);
    this.gl.deleteFramebuffer(framebuffer);

    // Convert back to tensor
    return wasmModule.WasmTensor.from_f32(result, new Uint32Array([shapeA[0], shapeB[1]]));
  }

  /**
   * Perform element-wise operations using WebGL
   * @param {Object} tensorA - First tensor
   * @param {Object} tensorB - Second tensor
   * @param {string} operation - Operation type ('add', 'sub', 'mul', 'div')
   * @returns {Object} Result tensor
   */
  async elementWise(tensorA, tensorB, operation) {
    if (!this.supported) {
      throw new Error('WebGL backend not initialized');
    }

    const program = this.programs.get('elementwise');
    this.gl.useProgram(program);

    // Get tensor data and shapes
    const dataA = await tensorA.to_js_array();
    const dataB = await tensorB.to_js_array();
    const shape = await tensorA.shape();

    // Create textures
    const width = shape[shape.length - 1];
    const height = Math.ceil(dataA.length / width);
    const textureA = this.createTexture(new Float32Array(dataA), width, height);
    const textureB = this.createTexture(new Float32Array(dataB), width, height);

    // Set uniforms
    const operationMap = { 'add': 0, 'sub': 1, 'mul': 2, 'div': 3 };
    this.gl.uniform1i(this.gl.getUniformLocation(program, 'u_textureA'), 0);
    this.gl.uniform1i(this.gl.getUniformLocation(program, 'u_textureB'), 1);
    this.gl.uniform1i(this.gl.getUniformLocation(program, 'u_operation'), operationMap[operation] || 0);

    // Bind textures
    this.gl.activeTexture(this.gl.TEXTURE0);
    this.gl.bindTexture(this.gl.TEXTURE_2D, textureA);
    this.gl.activeTexture(this.gl.TEXTURE1);
    this.gl.bindTexture(this.gl.TEXTURE_2D, textureB);

    // Setup framebuffer
    const { framebuffer, texture: outputTexture } = this.createFramebuffer(width, height);

    // Render
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, framebuffer);
    this.gl.viewport(0, 0, width, height);
    this.renderQuad(program);

    // Read result
    const result = this.readFramebuffer(width, height);

    // Cleanup
    this.gl.deleteTexture(textureA);
    this.gl.deleteTexture(textureB);
    this.gl.deleteTexture(outputTexture);
    this.gl.deleteFramebuffer(framebuffer);

    return wasmModule.WasmTensor.from_f32(result.slice(0, dataA.length), new Uint32Array(shape));
  }

  /**
   * Apply activation function using WebGL
   * @param {Object} tensor - Input tensor
   * @param {string} activation - Activation type ('relu', 'sigmoid', 'tanh', 'gelu')
   * @returns {Object} Result tensor
   */
  async activation(tensor, activation) {
    if (!this.supported) {
      throw new Error('WebGL backend not initialized');
    }

    const program = this.programs.get('activation');
    this.gl.useProgram(program);

    // Get tensor data and shape
    const data = await tensor.to_js_array();
    const shape = await tensor.shape();

    // Create texture
    const width = shape[shape.length - 1];
    const height = Math.ceil(data.length / width);
    const texture = this.createTexture(new Float32Array(data), width, height);

    // Set uniforms
    const activationMap = { 'relu': 0, 'sigmoid': 1, 'tanh': 2, 'gelu': 3 };
    this.gl.uniform1i(this.gl.getUniformLocation(program, 'u_texture'), 0);
    this.gl.uniform1i(this.gl.getUniformLocation(program, 'u_activation'), activationMap[activation] || 0);

    // Bind texture
    this.gl.activeTexture(this.gl.TEXTURE0);
    this.gl.bindTexture(this.gl.TEXTURE_2D, texture);

    // Setup framebuffer
    const { framebuffer, texture: outputTexture } = this.createFramebuffer(width, height);

    // Render
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, framebuffer);
    this.gl.viewport(0, 0, width, height);
    this.renderQuad(program);

    // Read result
    const result = this.readFramebuffer(width, height);

    // Cleanup
    this.gl.deleteTexture(texture);
    this.gl.deleteTexture(outputTexture);
    this.gl.deleteFramebuffer(framebuffer);

    return wasmModule.WasmTensor.from_f32(result.slice(0, data.length), new Uint32Array(shape));
  }

  /**
   * Create framebuffer for output
   * @param {number} width - Buffer width
   * @param {number} height - Buffer height
   * @returns {Object} Framebuffer and texture
   */
  createFramebuffer(width, height) {
    const framebuffer = this.gl.createFramebuffer();
    const texture = this.gl.createTexture();

    this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);

    const format = this.gl.RGBA;
    const type = this.extensions.floatTextures ? this.gl.FLOAT : this.gl.UNSIGNED_BYTE;
    this.gl.texImage2D(this.gl.TEXTURE_2D, 0, format, width, height, 0, format, type, null);

    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, framebuffer);
    this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, texture, 0);

    return { framebuffer, texture };
  }

  /**
   * Render full-screen quad
   * @param {WebGLProgram} program - Shader program
   */
  renderQuad(program) {
    // Setup attributes
    const positionLocation = this.gl.getAttribLocation(program, 'a_position');
    const texCoordLocation = this.gl.getAttribLocation(program, 'a_texCoord');

    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.get('quad'));
    
    this.gl.enableVertexAttribArray(positionLocation);
    this.gl.vertexAttribPointer(positionLocation, 2, this.gl.FLOAT, false, 16, 0);
    
    this.gl.enableVertexAttribArray(texCoordLocation);
    this.gl.vertexAttribPointer(texCoordLocation, 2, this.gl.FLOAT, false, 16, 8);

    // Draw
    this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
  }

  /**
   * Read framebuffer data
   * @param {number} width - Buffer width
   * @param {number} height - Buffer height
   * @returns {Float32Array} Buffer data
   */
  readFramebuffer(width, height) {
    if (this.extensions.floatTextures) {
      const buffer = new Float32Array(width * height * 4);
      this.gl.readPixels(0, 0, width, height, this.gl.RGBA, this.gl.FLOAT, buffer);
      return buffer;
    } 
      const buffer = new Uint8Array(width * height * 4);
      this.gl.readPixels(0, 0, width, height, this.gl.RGBA, this.gl.UNSIGNED_BYTE, buffer);
      // Convert back to float
      const floatBuffer = new Float32Array(width * height * 4);
      for (let i = 0; i < buffer.length; i++) {
        floatBuffer[i] = buffer[i] / 255.0;
      }
      return floatBuffer;
    
  }

  /**
   * Check if WebGL backend is supported
   * @returns {boolean} Support status
   */
  isSupported() {
    return this.supported;
  }

  /**
   * Get WebGL information
   * @returns {Object} WebGL info
   */
  getInfo() {
    if (!this.gl) return null;

    return {
      vendor: this.gl.getParameter(this.gl.VENDOR),
      renderer: this.gl.getParameter(this.gl.RENDERER),
      version: this.gl.getParameter(this.gl.VERSION),
      shadingLanguageVersion: this.gl.getParameter(this.gl.SHADING_LANGUAGE_VERSION),
      maxTextureSize: this.maxTextureSize,
      extensions: Object.keys(this.extensions).filter(key => this.extensions[key])
    };
  }

  /**
   * Cleanup resources
   */
  dispose() {
    if (this.gl) {
      // Delete programs
      for (const program of this.programs.values()) {
        this.gl.deleteProgram(program);
      }
      this.programs.clear();

      // Delete buffers
      for (const buffer of this.buffers.values()) {
        this.gl.deleteBuffer(buffer);
      }
      this.buffers.clear();

      // Delete textures
      for (const texture of this.textures.values()) {
        this.gl.deleteTexture(texture);
      }
      this.textures.clear();

      // Delete framebuffers
      for (const framebuffer of this.framebuffers.values()) {
        this.gl.deleteFramebuffer(framebuffer);
      }
      this.framebuffers.clear();
    }

    this.gl = null;
    this.supported = false;
  }
}

/**
 * Create and initialize WebGL backend
 * @param {HTMLCanvasElement} canvas - Canvas element (optional)
 * @returns {Promise<WebGLBackend>} Initialized backend
 */
export async function createWebGLBackend(canvas = null) {
  const backend = new WebGLBackend();
  const success = await backend.initialize(canvas);
  
  if (!success) {
    throw new Error('Failed to initialize WebGL backend');
  }
  
  return backend;
}