# Agentic Supply Chain Analytics (Olist)

This project is an agent-driven analytics system for e-commerce supply chain data built on the Olist dataset.

The system allows users to ask natural-language business questions (demand, volatility, categories, trends) while ensuring all numeric results are computed deterministically using Pandas rather than guessed by a language model.

The language model is used only for reasoning, planning, and explanation.

---

## Objective

Build a reliable analytics assistant that:
- understands business questions in natural language
- decides what analysis needs to be performed
- executes joins and aggregations exactly in Python
- explains results using verified outputs

The model never receives raw datasets and never computes numbers itself.

---

## Dataset

The project uses the Olist Brazilian E-commerce Dataset:

- olist_orders_dataset.csv
- olist_order_items_dataset.csv
- olist_products_dataset.csv
- product_category_name_translation.csv

### Demand definition
- One row in `order_items` represents one unit sold
- Demand is calculated using delivered orders by default

---

## What the Agent Can Answer

Examples of supported questions:

- Which products have the highest demand?
- Which product category has the highest demand volatility?
- Give me a demand summary for a specific product ID
- Show daily demand trend for a product
- Which categories are most unpredictable?
- Classify products as smooth, erratic, intermittent, or lumpy

---

## Architecture

# Agentic Supply Chain Analytics (Olist)

This project is an agent-driven analytics system for e-commerce supply chain data built on the Olist dataset.

The system allows users to ask natural-language business questions (demand, volatility, categories, trends) while ensuring all numeric results are computed deterministically using Pandas rather than guessed by a language model.

The language model is used only for reasoning, planning, and explanation.

---

## Objective

Build a reliable analytics assistant that:
- understands business questions in natural language
- decides what analysis needs to be performed
- executes joins and aggregations exactly in Python
- explains results using verified outputs

The model never receives raw datasets and never computes numbers itself.

---

## Dataset

The project uses the Olist Brazilian E-commerce Dataset:

- olist_orders_dataset.csv
- olist_order_items_dataset.csv
- olist_products_dataset.csv
- product_category_name_translation.csv

### Demand definition
- One row in `order_items` represents one unit sold
- Demand is calculated using delivered orders by default

---

## What the Agent Can Answer

Examples of supported questions:

- Which products have the highest demand?
- Which product category has the highest demand volatility?
- Give me a demand summary for a specific product ID
- Show daily demand trend for a product
- Which categories are most unpredictable?
- Classify products as smooth, erratic, intermittent, or lumpy

---

## Architecture

User Question
↓
LLM (Gemini)
↓
Tool Selection
↓
Pandas Computation
↓
Structured Result
↓
LLM Explanation

### Key properties
- Raw CSV files are never sent to the language model
- No numeric data is embedded or vectorized
- Only small computed results are passed to the model

---
