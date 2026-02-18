# Fashion Recommendation Engine

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Fashion Recommender Chatbot                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │   User      │    │   LLM Layer      │    │  Rule Engine  │  │
│  │   Input     │───▶│   (OpenAI GPT)   │───▶│  (Deterministic)│ │
│  └─────────────┘    └──────────────────┘    └───────────────┘  │
│                          │                            │           │
│                          ▼                            ▼           │
│                 ┌──────────────────┐    ┌───────────────────┐  │
│                 │  File Search      │    │  Outfit Assembler │  │
│                 │  (Vector Store)   │    │  (Rule-based)     │  │
│                 └──────────────────┘    └───────────────────┘  │
│                          │                            │           │
│                          ▼                            ▼           │
│                 ┌───────────────────────────────────────────┐   │
│                 │         Fashionpedia Taxonomy                │   │
│                 │   (Clean Metadata & Category Hierarchy)     │   │
│                 └───────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Fashionpedia Taxonomy
- Hierarchical category structure
- Attributes: color, style, season, occasion
- Compatibility rules
- Color matching matrix

### 2. File Search (RAG)
- Indexed fashion items descriptions
- Vector similarity search
- Context retrieval for LLM

### 3. Rule Layer
- Deterministic outfit assembler
- Color harmony rules
- Style compatibility
- Weather/season appropriateness
- Formality matching

### 4. LLM Integration
- Natural language understanding
- Personalized recommendations
- Style advice
- outfit explanations

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your-key-here"

# Run the chatbot
python src/chatbot.py

# Run outfit generation
python src/outfit_generator.py --occasion "casual" --weather "spring"
```
