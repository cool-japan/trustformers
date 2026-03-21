<!--
  TensorVisualization.svelte
  
  Interactive tensor visualization component with multiple display modes.
-->

<script>
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import { writable, derived } from 'svelte/store';
  import { 
    getTensorData, 
    createTensorSubscription,
    tensorVisualization 
  } from '../svelte_bindings.js';

  // Props
  export let tensorId = null;
  export let width = 300;
  export let height = 300;
  export let visualizationType = 'heatmap'; // 'heatmap', 'bar', 'line', 'histogram'
  export let colorScheme = 'viridis'; // 'viridis', 'plasma', 'coolwarm', 'grayscale'
  export let showControls = true;
  export let showStats = true;
  export let maxDisplayElements = 1000;
  export let updateInterval = 100; // ms

  // Event dispatcher
  const dispatch = createEventDispatcher();

  // State
  let canvasElement;
  let ctx;
  let animationFrame;
  let lastUpdateTime = 0;
  let isHovering = false;
  let hoverInfo = null;

  // Reactive stores
  const tensorSubscription = tensorId ? createTensorSubscription(tensorId) : writable(null);
  
  const tensorStats = derived(tensorSubscription, ($tensor) => {
    if (!$tensor?.data) return null;
    
    const data = Array.from($tensor.data);
    const sortedData = [...data].sort((a, b) => a - b);
    
    return {
      min: sortedData[0] || 0,
      max: sortedData[sortedData.length - 1] || 0,
      mean: data.reduce((sum, val) => sum + val, 0) / data.length || 0,
      median: sortedData[Math.floor(sortedData.length / 2)] || 0,
      std: Math.sqrt(data.reduce((sum, val) => sum + Math.pow(val - (data.reduce((s, v) => s + v, 0) / data.length), 2), 0) / data.length) || 0,
      zeros: data.filter(val => val === 0).length,
      nonZeros: data.filter(val => val !== 0).length,
      shape: $tensor.shape || [],
      dtype: $tensor.dtype || 'unknown',
      memoryUsage: $tensor.memoryUsage || 0
    };
  });

  // Color schemes
  const colorSchemes = {
    viridis: [
      [68, 1, 84],
      [72, 40, 120],
      [62, 74, 137],
      [49, 104, 142],
      [38, 130, 142],
      [31, 158, 137],
      [53, 183, 121],
      [109, 205, 89],
      [180, 222, 44],
      [253, 231, 37]
    ],
    plasma: [
      [13, 8, 135],
      [75, 3, 161],
      [125, 3, 168],
      [168, 34, 150],
      [199, 77, 125],
      [220, 124, 93],
      [234, 171, 53],
      [240, 219, 10]
    ],
    coolwarm: [
      [59, 76, 192],
      [144, 178, 254],
      [220, 220, 220],
      [245, 156, 125],
      [180, 4, 38]
    ],
    grayscale: [
      [0, 0, 0],
      [64, 64, 64],
      [128, 128, 128],
      [192, 192, 192],
      [255, 255, 255]
    ]
  };

  onMount(() => {
    if (canvasElement) {
      ctx = canvasElement.getContext('2d');
      startVisualization();
    }
  });

  onDestroy(() => {
    stopVisualization();
  });

  function startVisualization() {
    const update = () => {
      const now = performance.now();
      if (now - lastUpdateTime >= updateInterval) {
        renderVisualization();
        lastUpdateTime = now;
      }
      animationFrame = requestAnimationFrame(update);
    };
    update();
  }

  function stopVisualization() {
    if (animationFrame) {
      cancelAnimationFrame(animationFrame);
    }
  }

  function renderVisualization() {
    if (!ctx || !tensorId) return;

    try {
      const tensor = getTensorData(tensorId);
      if (!tensor?.data) return;

      // Clear canvas
      ctx.clearRect(0, 0, width, height);

      switch (visualizationType) {
        case 'heatmap':
          renderHeatmap(tensor);
          break;
        case 'bar':
          renderBarChart(tensor);
          break;
        case 'line':
          renderLineChart(tensor);
          break;
        case 'histogram':
          renderHistogram(tensor);
          break;
        default:
          renderHeatmap(tensor);
      }

      dispatch('render', { tensor, timestamp: performance.now() });
    } catch (error) {
      console.error('Visualization render error:', error);
      dispatch('error', { error });
    }
  }

  function renderHeatmap(tensor) {
    if (tensor.shape.length !== 2) {
      renderFallbackVisualization(tensor);
      return;
    }

    const [rows, cols] = tensor.shape;
    const cellWidth = width / cols;
    const cellHeight = height / rows;

    // Calculate value range for normalization
    const data = Array.from(tensor.data);
    const minVal = Math.min(...data);
    const maxVal = Math.max(...data);
    const range = maxVal - minVal || 1;

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const value = tensor.data[i * cols + j];
        const normalized = (value - minVal) / range;
        const color = getColor(normalized, colorScheme);

        ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
        ctx.fillRect(j * cellWidth, i * cellHeight, cellWidth, cellHeight);

        // Add value text for small matrices
        if (rows <= 10 && cols <= 10) {
          ctx.fillStyle = normalized > 0.5 ? 'white' : 'black';
          ctx.font = `${Math.min(cellWidth, cellHeight) / 3}px Arial`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(
            value.toFixed(2),
            j * cellWidth + cellWidth / 2,
            i * cellHeight + cellHeight / 2
          );
        }
      }
    }
  }

  function renderBarChart(tensor) {
    const data = Array.from(tensor.data.slice(0, maxDisplayElements));
    const maxVal = Math.max(...data.map(Math.abs));
    const barWidth = width / data.length;

    ctx.strokeStyle = '#ccc';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    data.forEach((value, index) => {
      const barHeight = Math.abs(value) / maxVal * (height / 2);
      const x = index * barWidth;
      const y = value >= 0 ? height / 2 - barHeight : height / 2;

      ctx.fillStyle = value >= 0 ? '#4CAF50' : '#F44336';
      ctx.fillRect(x, y, barWidth - 1, barHeight);
    });
  }

  function renderLineChart(tensor) {
    const data = Array.from(tensor.data.slice(0, maxDisplayElements));
    const minVal = Math.min(...data);
    const maxVal = Math.max(...data);
    const range = maxVal - minVal || 1;

    ctx.strokeStyle = '#2196F3';
    ctx.lineWidth = 2;
    ctx.beginPath();

    data.forEach((value, index) => {
      const x = (index / (data.length - 1)) * width;
      const y = height - ((value - minVal) / range) * height;
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    // Add points
    ctx.fillStyle = '#1976D2';
    data.forEach((value, index) => {
      const x = (index / (data.length - 1)) * width;
      const y = height - ((value - minVal) / range) * height;
      
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    });
  }

  function renderHistogram(tensor) {
    const data = Array.from(tensor.data);
    const bins = 20;
    const minVal = Math.min(...data);
    const maxVal = Math.max(...data);
    const binWidth = (maxVal - minVal) / bins;

    // Calculate histogram
    const histogram = new Array(bins).fill(0);
    data.forEach(value => {
      const binIndex = Math.min(Math.floor((value - minVal) / binWidth), bins - 1);
      histogram[binIndex]++;
    });

    const maxCount = Math.max(...histogram);
    const barWidth = width / bins;

    // Draw histogram bars
    histogram.forEach((count, index) => {
      const barHeight = (count / maxCount) * height;
      const x = index * barWidth;
      const y = height - barHeight;

      ctx.fillStyle = '#9C27B0';
      ctx.fillRect(x, y, barWidth - 1, barHeight);

      // Add count labels
      if (count > 0) {
        ctx.fillStyle = 'black';
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(count.toString(), x + barWidth / 2, y - 5);
      }
    });
  }

  function renderFallbackVisualization(tensor) {
    const data = Array.from(tensor.data.slice(0, maxDisplayElements));
    renderBarChart(tensor);
    
    // Add shape info
    ctx.fillStyle = 'black';
    ctx.font = '12px Arial';
    ctx.textAlign = 'left';
    ctx.fillText(`Shape: [${tensor.shape.join(', ')}]`, 10, 20);
  }

  function getColor(normalized, scheme) {
    const colors = colorSchemes[scheme] || colorSchemes.viridis;
    const index = Math.floor(normalized * (colors.length - 1));
    const nextIndex = Math.min(index + 1, colors.length - 1);
    const t = (normalized * (colors.length - 1)) - index;

    const color1 = colors[index];
    const color2 = colors[nextIndex];

    return [
      Math.round(color1[0] + (color2[0] - color1[0]) * t),
      Math.round(color1[1] + (color2[1] - color1[1]) * t),
      Math.round(color1[2] + (color2[2] - color1[2]) * t)
    ];
  }

  function handleCanvasHover(event) {
    if (!tensorId) return;

    const rect = canvasElement.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    try {
      const tensor = getTensorData(tensorId);
      if (!tensor?.data || tensor.shape.length !== 2) return;

      const [rows, cols] = tensor.shape;
      const cellWidth = width / cols;
      const cellHeight = height / rows;

      const col = Math.floor(x / cellWidth);
      const row = Math.floor(y / cellHeight);

      if (row >= 0 && row < rows && col >= 0 && col < cols) {
        const index = row * cols + col;
        const value = tensor.data[index];

        hoverInfo = {
          x: col,
          y: row,
          value: value,
          index: index,
          mouseX: event.clientX,
          mouseY: event.clientY
        };
      } else {
        hoverInfo = null;
      }
    } catch (error) {
      hoverInfo = null;
    }
  }

  function handleCanvasLeave() {
    hoverInfo = null;
    isHovering = false;
  }

  // Reactive updates
  $: if (tensorId && canvasElement) {
    renderVisualization();
  }

  $: if (visualizationType || colorScheme) {
    renderVisualization();
  }
</script>

<div class="tensor-visualization">
  {#if showControls}
    <div class="controls">
      <label>
        Type:
        <select bind:value={visualizationType}>
          <option value="heatmap">Heatmap</option>
          <option value="bar">Bar Chart</option>
          <option value="line">Line Chart</option>
          <option value="histogram">Histogram</option>
        </select>
      </label>

      <label>
        Colors:
        <select bind:value={colorScheme}>
          <option value="viridis">Viridis</option>
          <option value="plasma">Plasma</option>
          <option value="coolwarm">Cool-Warm</option>
          <option value="grayscale">Grayscale</option>
        </select>
      </label>
    </div>
  {/if}

  <div class="canvas-container">
    <canvas
      bind:this={canvasElement}
      {width}
      {height}
      on:mousemove={handleCanvasHover}
      on:mouseenter={() => isHovering = true}
      on:mouseleave={handleCanvasLeave}
      role="img"
      aria-label="Tensor visualization"
    ></canvas>

    {#if hoverInfo}
      <div 
        class="tooltip" 
        style="left: {hoverInfo.mouseX + 10}px; top: {hoverInfo.mouseY - 30}px;"
      >
        [{hoverInfo.x}, {hoverInfo.y}]: {hoverInfo.value.toFixed(4)}
      </div>
    {/if}
  </div>

  {#if showStats && $tensorStats}
    <div class="stats">
      <div class="stats-grid">
        <div class="stat">
          <span class="label">Shape:</span>
          <span class="value">[{$tensorStats.shape.join(', ')}]</span>
        </div>
        <div class="stat">
          <span class="label">Min:</span>
          <span class="value">{$tensorStats.min.toFixed(4)}</span>
        </div>
        <div class="stat">
          <span class="label">Max:</span>
          <span class="value">{$tensorStats.max.toFixed(4)}</span>
        </div>
        <div class="stat">
          <span class="label">Mean:</span>
          <span class="value">{$tensorStats.mean.toFixed(4)}</span>
        </div>
        <div class="stat">
          <span class="label">Std:</span>
          <span class="value">{$tensorStats.std.toFixed(4)}</span>
        </div>
        <div class="stat">
          <span class="label">Memory:</span>
          <span class="value">{($tensorStats.memoryUsage / 1024).toFixed(1)} KB</span>
        </div>
      </div>
    </div>
  {/if}

  {#if !tensorId}
    <div class="no-tensor">
      <p>No tensor selected for visualization</p>
    </div>
  {/if}
</div>

<style>
  .tensor-visualization {
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 1rem;
    background: white;
    font-family: 'Segoe UI', Arial, sans-serif;
  }

  .controls {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
  }

  .controls label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.9rem;
  }

  .controls select {
    padding: 0.25rem;
    border: 1px solid #ccc;
    border-radius: 3px;
  }

  .canvas-container {
    position: relative;
    display: inline-block;
    border: 1px solid #eee;
    border-radius: 3px;
  }

  canvas {
    display: block;
    cursor: crosshair;
  }

  .tooltip {
    position: fixed;
    background: rgba(0, 0, 0, 0.8);
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
    font-size: 0.8rem;
    font-family: monospace;
    z-index: 1000;
    pointer-events: none;
    white-space: nowrap;
  }

  .stats {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid #eee;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 0.5rem;
  }

  .stat {
    display: flex;
    justify-content: space-between;
    padding: 0.25rem;
    background: #f9f9f9;
    border-radius: 3px;
  }

  .stat .label {
    font-weight: bold;
    color: #666;
  }

  .stat .value {
    font-family: monospace;
    color: #333;
  }

  .no-tensor {
    text-align: center;
    color: #999;
    padding: 2rem;
    border: 2px dashed #ddd;
    border-radius: 4px;
  }

  /* Responsive design */
  @media (max-width: 768px) {
    .controls {
      flex-direction: column;
    }
    
    .stats-grid {
      grid-template-columns: 1fr;
    }
  }

  /* Dark mode support */
  @media (prefers-color-scheme: dark) {
    .tensor-visualization {
      background: #2d2d2d;
      border-color: #555;
      color: white;
    }
    
    .stat {
      background: #3d3d3d;
    }
    
    .controls select {
      background: #3d3d3d;
      color: white;
      border-color: #555;
    }
  }
</style>