<!--
  TensorOperations.svelte
  
  Interactive tensor operations component with visual operation builder.
-->

<script>
  import { onMount, createEventDispatcher } from 'svelte';
  import { writable, derived } from 'svelte/store';
  import { 
    createTensor, 
    deleteTensor, 
    getTensorData, 
    addTensors, 
    matrixMultiply,
    tensorStore,
    measureOperation,
    createReactiveTensorOp,
    createErrorHandler
  } from '../svelte_bindings.js';

  // Props
  export let showCreation = true;
  export let showOperations = true;
  export let showHistory = true;
  export let maxHistoryItems = 20;

  // Event dispatcher
  const dispatch = createEventDispatcher();

  // Error handling
  const errorHandler = createErrorHandler('TensorOperations');

  // Local state
  let selectedTensors = [];
  let operationHistory = writable([]);
  let isProcessing = false;

  // Tensor creation form
  let createForm = {
    shape: '3,3',
    fillType: 'zeros', // 'zeros', 'ones', 'random', 'custom'
    customValues: '',
    name: ''
  };

  // Operation selection
  let selectedOperation = 'add';
  let operationParams = {};

  // Available operations
  const operations = {
    add: {
      name: 'Addition',
      description: 'Element-wise addition of two tensors',
      inputCount: 2,
      params: []
    },
    matmul: {
      name: 'Matrix Multiplication',
      description: 'Matrix multiplication of two 2D tensors',
      inputCount: 2,
      params: []
    },
    softmax: {
      name: 'Softmax',
      description: 'Softmax activation function',
      inputCount: 1,
      params: [
        { name: 'axis', type: 'number', default: -1, description: 'Axis along which to compute softmax' }
      ]
    },
    relu: {
      name: 'ReLU',
      description: 'Rectified Linear Unit activation',
      inputCount: 1,
      params: []
    },
    transpose: {
      name: 'Transpose',
      description: 'Transpose tensor dimensions',
      inputCount: 1,
      params: [
        { name: 'axes', type: 'text', default: '1,0', description: 'New axis order (comma-separated)' }
      ]
    }
  };

  // Derived stores
  const availableTensors = derived(tensorStore, ($tensorStore) => {
    return Array.from($tensorStore.entries()).map(([id, info]) => ({
      id,
      ...info,
      displayName: info.name || `Tensor ${id}`
    }));
  });

  const canPerformOperation = derived(
    [availableTensors, tensorStore],
    ([$availableTensors, $tensorStore]) => {
      const operation = operations[selectedOperation];
      if (!operation) return false;
      
      return selectedTensors.length >= operation.inputCount &&
             selectedTensors.every(id => $tensorStore.has(id));
    }
  );

  onMount(() => {
    // Initialize operation parameters
    updateOperationParams();
  });

  function updateOperationParams() {
    const operation = operations[selectedOperation];
    if (!operation) return;

    operationParams = {};
    operation.params.forEach(param => {
      operationParams[param.name] = param.default;
    });
  }

  async function createNewTensor() {
    try {
      isProcessing = true;
      
      // Parse shape
      const shape = createForm.shape.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
      if (shape.length === 0) {
        throw new Error('Invalid shape format');
      }

      // Generate data based on fill type
      let data = null;
      const size = shape.reduce((a, b) => a * b, 1);

      switch (createForm.fillType) {
        case 'zeros':
          data = new Array(size).fill(0);
          break;
        case 'ones':
          data = new Array(size).fill(1);
          break;
        case 'random':
          data = Array.from({ length: size }, () => Math.random() * 2 - 1);
          break;
        case 'custom':
          if (createForm.customValues.trim()) {
            const values = createForm.customValues.split(',').map(s => parseFloat(s.trim()));
            if (values.length !== size) {
              throw new Error(`Expected ${size} values, got ${values.length}`);
            }
            data = values;
          }
          break;
      }

      const tensorId = await measureOperation('createTensor', createTensor)(shape, data);
      
      // Add to history
      addToHistory({
        type: 'creation',
        operation: `Create ${createForm.fillType} tensor`,
        inputs: [],
        output: tensorId,
        shape,
        timestamp: new Date(),
        params: { fillType: createForm.fillType }
      });

      // Reset form
      createForm = {
        shape: '3,3',
        fillType: 'zeros',
        customValues: '',
        name: ''
      };

      dispatch('tensor-created', { tensorId, shape, data });
      
    } catch (error) {
      errorHandler.handle(error, 'tensor creation');
      dispatch('error', { error, context: 'tensor-creation' });
    } finally {
      isProcessing = false;
    }
  }

  async function performOperation() {
    if (!$canPerformOperation) return;

    try {
      isProcessing = true;
      
      const operation = operations[selectedOperation];
      const inputs = selectedTensors.slice(0, operation.inputCount);
      
      let resultId;
      let operationFn;

      switch (selectedOperation) {
        case 'add':
          operationFn = () => addTensors(inputs[0], inputs[1]);
          break;
        case 'matmul':
          operationFn = () => matrixMultiply(inputs[0], inputs[1]);
          break;
        case 'softmax':
          operationFn = () => {
            const instance = getTensorData(inputs[0]);
            // In a real implementation, this would call the WASM softmax function
            return createTensor(instance.shape, Array.from(instance.data));
          };
          break;
        case 'relu':
          operationFn = () => {
            const instance = getTensorData(inputs[0]);
            const data = Array.from(instance.data).map(x => Math.max(0, x));
            return createTensor(instance.shape, data);
          };
          break;
        case 'transpose':
          operationFn = () => {
            const instance = getTensorData(inputs[0]);
            // Simple transpose for 2D tensors
            if (instance.shape.length === 2) {
              const [rows, cols] = instance.shape;
              const transposed = new Array(rows * cols);
              for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                  transposed[j * rows + i] = instance.data[i * cols + j];
                }
              }
              return createTensor([cols, rows], transposed);
            }
            return createTensor(instance.shape, Array.from(instance.data));
          };
          break;
        default:
          throw new Error(`Unknown operation: ${selectedOperation}`);
      }

      resultId = await measureOperation(selectedOperation, operationFn)();

      // Add to history
      addToHistory({
        type: 'operation',
        operation: operation.name,
        inputs: [...inputs],
        output: resultId,
        timestamp: new Date(),
        params: { ...operationParams }
      });

      // Clear selection
      selectedTensors = [];

      dispatch('operation-completed', { 
        operation: selectedOperation, 
        inputs, 
        output: resultId,
        params: operationParams
      });

    } catch (error) {
      errorHandler.handle(error, 'tensor operation');
      dispatch('error', { error, context: 'tensor-operation' });
    } finally {
      isProcessing = false;
    }
  }

  function addToHistory(item) {
    operationHistory.update(history => {
      const newHistory = [item, ...history];
      return newHistory.slice(0, maxHistoryItems);
    });
  }

  function deleteTensorWithConfirmation(tensorId) {
    if (confirm('Are you sure you want to delete this tensor?')) {
      deleteTensor(tensorId);
      selectedTensors = selectedTensors.filter(id => id !== tensorId);
      
      addToHistory({
        type: 'deletion',
        operation: 'Delete tensor',
        inputs: [tensorId],
        output: null,
        timestamp: new Date(),
        params: {}
      });

      dispatch('tensor-deleted', { tensorId });
    }
  }

  function clearHistory() {
    operationHistory.set([]);
  }

  function selectTensor(tensorId) {
    const operation = operations[selectedOperation];
    if (!operation) return;

    if (selectedTensors.includes(tensorId)) {
      selectedTensors = selectedTensors.filter(id => id !== tensorId);
    } else if (selectedTensors.length < operation.inputCount) {
      selectedTensors = [...selectedTensors, tensorId];
    } else {
      // Replace the first selected tensor
      selectedTensors = [selectedTensors[1], tensorId].filter(Boolean);
    }
  }

  function formatTimestamp(date) {
    return date.toLocaleTimeString();
  }

  function getTensorInfo(tensorId) {
    try {
      const data = getTensorData(tensorId);
      return data ? {
        shape: data.shape,
        size: data.data.length,
        memory: data.data.byteLength
      } : null;
    } catch {
      return null;
    }
  }

  // Reactive statements
  $: if (selectedOperation) {
    updateOperationParams();
    selectedTensors = [];
  }
</script>

<div class="tensor-operations">
  <!-- Tensor Creation Section -->
  {#if showCreation}
    <section class="creation-section">
      <h3>Create Tensor</h3>
      <form on:submit|preventDefault={createNewTensor} class="creation-form">
        <div class="form-row">
          <label>
            Shape (comma-separated):
            <input 
              type="text" 
              bind:value={createForm.shape} 
              placeholder="3,3"
              required
            />
          </label>
          
          <label>
            Fill Type:
            <select bind:value={createForm.fillType}>
              <option value="zeros">Zeros</option>
              <option value="ones">Ones</option>
              <option value="random">Random</option>
              <option value="custom">Custom Values</option>
            </select>
          </label>
        </div>

        {#if createForm.fillType === 'custom'}
          <div class="form-row">
            <label>
              Values (comma-separated):
              <textarea 
                bind:value={createForm.customValues} 
                placeholder="1,2,3,4,5,6,7,8,9"
                rows="3"
              ></textarea>
            </label>
          </div>
        {/if}

        <button type="submit" disabled={isProcessing} class="create-button">
          {isProcessing ? 'Creating...' : 'Create Tensor'}
        </button>
      </form>
    </section>
  {/if}

  <!-- Available Tensors Section -->
  <section class="tensors-section">
    <h3>Available Tensors ({$availableTensors.length})</h3>
    
    {#if $availableTensors.length === 0}
      <p class="no-tensors">No tensors available. Create one above.</p>
    {:else}
      <div class="tensor-list">
        {#each $availableTensors as tensor (tensor.id)}
          {@const tensorInfo = getTensorInfo(tensor.id)}
          <div 
            class="tensor-item" 
            class:selected={selectedTensors.includes(tensor.id)}
            on:click={() => selectTensor(tensor.id)}
            on:keydown={(e) => e.key === 'Enter' && selectTensor(tensor.id)}
            role="button"
            tabindex="0"
          >
            <div class="tensor-info">
              <strong>Tensor {tensor.id}</strong>
              {#if tensorInfo}
                <div class="tensor-details">
                  <span>Shape: [{tensorInfo.shape.join(', ')}]</span>
                  <span>Size: {tensorInfo.size}</span>
                  <span>Memory: {(tensorInfo.memory / 1024).toFixed(1)} KB</span>
                </div>
              {/if}
              <div class="tensor-meta">
                <span class="device">Device: {tensor.device}</span>
                <span class="created">Created: {formatTimestamp(tensor.createdAt)}</span>
              </div>
            </div>
            
            <button 
              class="delete-button"
              on:click|stopPropagation={() => deleteTensorWithConfirmation(tensor.id)}
              aria-label="Delete tensor {tensor.id}"
            >
              ×
            </button>
          </div>
        {/each}
      </div>
    {/if}
  </section>

  <!-- Operations Section -->
  {#if showOperations && $availableTensors.length > 0}
    <section class="operations-section">
      <h3>Tensor Operations</h3>
      
      <div class="operation-controls">
        <label>
          Operation:
          <select bind:value={selectedOperation}>
            {#each Object.entries(operations) as [key, op]}
              <option value={key}>{op.name}</option>
            {/each}
          </select>
        </label>

        {#if operations[selectedOperation]?.params.length > 0}
          <div class="operation-params">
            <h4>Parameters:</h4>
            {#each operations[selectedOperation].params as param}
              <label>
                {param.description}:
                {#if param.type === 'number'}
                  <input 
                    type="number" 
                    bind:value={operationParams[param.name]}
                  />
                {:else}
                  <input 
                    type="text" 
                    bind:value={operationParams[param.name]}
                  />
                {/if}
              </label>
            {/each}
          </div>
        {/if}
      </div>

      <div class="selection-info">
        <p>
          Select {operations[selectedOperation]?.inputCount || 0} tensor(s) for 
          <strong>{operations[selectedOperation]?.name}</strong>
        </p>
        <p class="operation-description">
          {operations[selectedOperation]?.description}
        </p>
        
        {#if selectedTensors.length > 0}
          <p class="selected-tensors">
            Selected: {selectedTensors.map(id => `Tensor ${id}`).join(', ')}
          </p>
        {/if}
      </div>

      <button 
        class="perform-operation"
        disabled={!$canPerformOperation || isProcessing}
        on:click={performOperation}
      >
        {isProcessing ? 'Processing...' : `Perform ${operations[selectedOperation]?.name}`}
      </button>
    </section>
  {/if}

  <!-- History Section -->
  {#if showHistory}
    <section class="history-section">
      <div class="history-header">
        <h3>Operation History</h3>
        <button on:click={clearHistory} class="clear-history">Clear</button>
      </div>
      
      {#if $operationHistory.length === 0}
        <p class="no-history">No operations performed yet.</p>
      {:else}
        <div class="history-list">
          {#each $operationHistory as item (item.timestamp)}
            <div class="history-item" class:creation={item.type === 'creation'}>
              <div class="history-operation">
                <strong>{item.operation}</strong>
                <span class="timestamp">{formatTimestamp(item.timestamp)}</span>
              </div>
              
              <div class="history-details">
                {#if item.inputs.length > 0}
                  <span>Inputs: {item.inputs.map(id => `Tensor ${id}`).join(', ')}</span>
                {/if}
                {#if item.output}
                  <span>Output: Tensor {item.output}</span>
                {/if}
                {#if Object.keys(item.params).length > 0}
                  <span>Params: {JSON.stringify(item.params)}</span>
                {/if}
              </div>
            </div>
          {/each}
        </div>
      {/if}
    </section>
  {/if}
</div>

<style>
  .tensor-operations {
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1.5rem;
    font-family: 'Segoe UI', Arial, sans-serif;
  }

  section {
    margin-bottom: 2rem;
  }

  section:last-child {
    margin-bottom: 0;
  }

  h3 {
    margin: 0 0 1rem 0;
    color: #333;
    border-bottom: 2px solid #eee;
    padding-bottom: 0.5rem;
  }

  /* Creation Form */
  .creation-form {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .form-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
  }

  .form-row label {
    flex: 1;
    min-width: 200px;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  input, select, textarea {
    padding: 0.5rem;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 0.9rem;
  }

  textarea {
    resize: vertical;
    font-family: monospace;
  }

  .create-button {
    background: #2196f3;
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1rem;
    align-self: flex-start;
  }

  .create-button:hover:not(:disabled) {
    background: #1976d2;
  }

  .create-button:disabled {
    background: #ccc;
    cursor: not-allowed;
  }

  /* Tensor List */
  .tensor-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .tensor-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem;
    border: 1px solid #ddd;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .tensor-item:hover {
    border-color: #2196f3;
    background: #f5f5f5;
  }

  .tensor-item.selected {
    border-color: #2196f3;
    background: #e3f2fd;
  }

  .tensor-info {
    flex: 1;
  }

  .tensor-details {
    display: flex;
    gap: 1rem;
    font-size: 0.8rem;
    color: #666;
    margin: 0.25rem 0;
  }

  .tensor-meta {
    display: flex;
    gap: 1rem;
    font-size: 0.7rem;
    color: #999;
  }

  .delete-button {
    background: #f44336;
    color: white;
    border: none;
    width: 30px;
    height: 30px;
    border-radius: 50%;
    cursor: pointer;
    font-size: 1.2rem;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .delete-button:hover {
    background: #d32f2f;
  }

  .no-tensors, .no-history {
    text-align: center;
    color: #999;
    padding: 2rem;
    border: 2px dashed #ddd;
    border-radius: 4px;
  }

  /* Operations */
  .operation-controls {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-bottom: 1rem;
  }

  .operation-params {
    padding: 1rem;
    background: #f9f9f9;
    border-radius: 4px;
  }

  .operation-params h4 {
    margin: 0 0 0.5rem 0;
  }

  .selection-info {
    background: #f0f8ff;
    padding: 1rem;
    border-radius: 4px;
    margin-bottom: 1rem;
  }

  .operation-description {
    font-style: italic;
    color: #666;
  }

  .selected-tensors {
    font-weight: bold;
    color: #2196f3;
  }

  .perform-operation {
    background: #4caf50;
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1rem;
    width: 100%;
  }

  .perform-operation:hover:not(:disabled) {
    background: #45a049;
  }

  .perform-operation:disabled {
    background: #ccc;
    cursor: not-allowed;
  }

  /* History */
  .history-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .clear-history {
    background: #ff9800;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.9rem;
  }

  .history-list {
    max-height: 300px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .history-item {
    padding: 0.75rem;
    background: #f9f9f9;
    border-radius: 4px;
    border-left: 4px solid #2196f3;
  }

  .history-item.creation {
    border-left-color: #4caf50;
  }

  .history-operation {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.25rem;
  }

  .timestamp {
    font-size: 0.8rem;
    color: #666;
  }

  .history-details {
    font-size: 0.8rem;
    color: #666;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  /* Responsive design */
  @media (max-width: 768px) {
    .form-row {
      flex-direction: column;
    }
    
    .tensor-details {
      flex-direction: column;
      gap: 0.25rem;
    }
    
    .operation-controls {
      gap: 0.5rem;
    }
  }
</style>