#!/bin/bash
# TrustformeRS Model Template Quick Start
# This script demonstrates how to use the model templates

echo "🚀 TrustformeRS Model Template Quick Start"
echo "========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found"
    exit 1
fi

# Display available templates
echo "📋 Available Templates:"
echo "  1. Transformer (BERT, GPT, T5, etc.)"
echo "  2. CNN (ResNet, EfficientNet, etc.)"
echo "  3. Custom (GNN, RNN, specialized architectures)"
echo ""

# Example 1: Generate a BERT model
echo "📝 Example 1: Generating a BERT model"
echo "-------------------------------------"
echo "Command:"
echo "  python3 generate_model.py --type transformer --name BERT --config examples/bert_config.json --output ../bert/"
echo ""
read -p "Press Enter to generate BERT model (or Ctrl+C to skip)..."

if [ -f "generate_model.py" ]; then
    python3 generate_model.py --type transformer --name BERT --config examples/bert_config.json --output ../bert/
    echo "✅ BERT model generated in ../bert/"
    echo ""
    echo "Generated files:"
    ls -la ../bert/ 2>/dev/null || echo "  (Output directory not accessible)"
fi

echo ""
echo "📝 Example 2: Interactive CNN Model Generation"
echo "---------------------------------------------"
echo "Command:"
echo "  python3 generate_model.py --type cnn --name MyResNet --output ../myresnet/ --interactive"
echo ""
echo "This will prompt you for configuration options."
echo ""

echo "📝 Example 3: Custom Model from Config"
echo "-------------------------------------"
echo "Command:"
echo "  python3 generate_model.py --type custom --name GraphAttentionNetwork --config examples/gnn_config.json --output ../gat/"
echo ""

echo "📚 Next Steps:"
echo "-------------"
echo "1. Review generated model in the output directory"
echo "2. Implement model-specific logic in model.rs"
echo "3. Add specialized heads (ForSequenceClassification, etc.)"
echo "4. Write comprehensive tests"
echo "5. Add documentation and examples"
echo ""

echo "📖 Resources:"
echo "- Template Guide: TEMPLATE_GUIDE.md"
echo "- Example Configs: examples/"
echo "- Model Tutorial: ../../docs/tutorials/model_implementation.md"
echo ""

echo "💡 Tips:"
echo "- Start with an example config and modify it"
echo "- Use --interactive for guided configuration"
echo "- Check generated code for TODO comments"
echo "- Run 'cargo check' to verify syntax"
echo ""

echo "Happy modeling! 🎉"