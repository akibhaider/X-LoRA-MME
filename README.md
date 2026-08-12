[![Watch Demo](https://YOUR-THUMBNAIL-URL.png)](https://drive.google.com/file/d/11b-wPbunSQw7JsNYQGRQ52nk94_LEtRV/view)

# LoRA-MME: Multi-Model Ensemble of LoRA-Tuned Encoders for Code Comment Classification

**X-LORA MME** is a multi-model ensemble architecture developed for the **NLBSE'26 Tool Competition**. It utilizes Parameter-Efficient Fine-Tuning (PEFT) to address the multi-label code comment classification challenge across **Java**, **Python**, and **Pharo**.

By combining the strengths of four distinct transformer encoders—**UniXcoder**, **CodeBERT**, **GraphCodeBERT**, and **CodeBERTa**—and fine-tuning them independently using Low-Rank Adaptation (LoRA), this tool maximizes classification performance while maintaining memory efficiency.

## Key Features

* **Multi-Model Ensemble**: Aggregates predictions from four specialized code encoders to capture diverse semantic and structural features:
    * **UniXcoder**: Handles cross-modal tasks and AST representations.
    * **CodeBERT**: Provides semantic alignment between natural language and code.
    * **GraphCodeBERT**: Captures data flow and semantic-level structure, crucial for categories like *Pointer* and *Usage*.
    * **CodeBERTa**: Offers complementary representations with lower computational overhead.
* **Parameter-Efficient Fine-Tuning**: Uses LoRA to fine-tune only ~4.5% of parameters (approx. 5.9M) per model, allowing training on consumer hardware (RTX 3090).
* **Learned Weighted Ensemble**: Instead of simple probability averaging, the model learns category-specific mixing weights to dynamically prioritize the most effective encoder for each comment type.
* **Threshold Optimization**: Implements per-category decision thresholds (ranging from 0.28 to 0.85) to address class imbalance and improve F1 scores for underrepresented categories.

## Repository Structure

* `Evaluation_run0.ipynb`: Evaluation notebook for the baseline model run.
* `Evaluation_run1_reduced_parameter.ipynb`: Evaluation notebook for the reduced-parameter run focusing on efficiency.
* `X_LoRA_MME.pdf`: The technical report detailing the architecture and results.

## Methodology

### Architecture
The architecture consists of four base models independently fine-tuned using LoRA adapters. The predictions are combined using a learned weight vector $W_c$ for each category $c$. The final probability is computed as:

$$P(c|x) = \sum_{m=1}^{4} w_{m,c} \cdot \sigma(z_{m,c})$$

### LoRA Configuration
LoRA adapters are injected into the `query`, `key`, `value`, and `dense` layers of the attention mechanism with the following hyperparameters:

* **Rank (r)**: 16
* **Alpha ($\alpha$)**: 32
* **Dropout**: 0.1

## Performance

The tool achieved an **F1 Weighted score of 0.7906** and a **Macro F1 of 0.6867** on the test set.

### Quantitative Analysis

| Metric | Score |
| :--- | :--- |
| **Weighted F1** | 0.7906 |
| **Macro F1** | 0.6867 |
| **Submission Score** | 41.20% |

*The submission score reflects the trade-off between the ensemble's high semantic accuracy and the computational cost of running four models.*

### Language-Specific Results

| Language | Macro F1 | Baseline F1 | Improvement |
| :--- | :--- | :--- | :--- |
| **Java** | 0.7445 | 0.7306 | +0.0139 |
| **Python** | 0.6296 | 0.5820 | +0.0476 |
| **Pharo** | 0.6668 | 0.6152 | +0.0516 |

## Authors

* **[Md Akib Haider](https://github.com/akibhaider)** (Islamic University of Technology)
* **[Ahsan Bulbul](https://github.com/ahsanbulbul00)** (Islamic University of Technology)
* **[Nafis Fuad Shahid](https://github.com/NafisFuadShahid)** (Islamic University of Technology)
* **Aimaan Ahmed** (Islamic University of Technology)
* **Mohammad Ishrak Abedin** (Islamic University of Technology)

## Acknowledgments

We thank the **NLBSE 2026** organizers and the **Islamic University of Technology (IUT)** for providing computing resources. We also thank **Syed Rifat Raiyan** and **Ajwad Abrar Mostofa** for their assistance.
