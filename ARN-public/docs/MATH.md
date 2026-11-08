# Mathematical Specification of the Aether Routing Network (ARN)

This document provides the formal mathematical formulation of the Aether Routing Network (ARN), covering the Factored Attention Routing (FAR) mechanism, the Linear-Chain Conditional Random Field (CRF) with hard BIO masking, loss normalization, and dynamic programming time/space complexities.

---

## 1. Factored Attention Routing (FAR)

Let $\mathbf{x} = (x_1, \dots, x_T)$ denote an input token sequence of length $T$, and $\mathbf{H} \in \mathbb{R}^{T \times d}$ denote the output representations from the bidirectional Transformer trunk (BERT-tiny, $d = 128$):
$$\mathbf{H} = \text{Encoder}(\mathbf{x})$$

Let $\mathcal{T} = \{t_1, t_2, \dots, t_K\}$ represent the closed tool registry containing $K$ predefined API tools ($K=18$). Each tool $k \in \{1, \dots, K\}$ is parameterized by a learned query embedding:
$$\mathbf{q}_k \in \mathbb{R}^{d}$$
collected into query matrix $\mathbf{Q} \in \mathbb{R}^{K \times d}$.

### 1.1 Cross-Attention Routing
To compute routing relevance, token representations are projected into key space:
$$\mathbf{K}_{\text{proj}} = \mathbf{H} \mathbf{W}_K, \quad \mathbf{W}_K \in \mathbb{R}^{d \times d}$$

The attention affinity matrix $\mathbf{S} \in \mathbb{R}^{K \times T}$ is computed via scaled dot-product:
$$S_{k, t} = \frac{\mathbf{q}_k^\top (\mathbf{K}_{\text{proj}})_t}{\sqrt{d}} + M_t^{\text{pad}}$$
where $M_t^{\text{pad}} = -\infty$ if token $t$ is a padding token, and $0$ otherwise.

The routing attention weights $\boldsymbol{\alpha}_k \in \mathbb{R}^T$ are obtained by softmax over token positions:
$$\alpha_{k, t} = \frac{\exp(S_{k, t})}{\sum_{t'=1}^T \exp(S_{k, t'})}$$

The context vector $\mathbf{c}_k \in \mathbb{R}^d$ for tool $k$ is the attention-weighted sum of token hidden states:
$$\mathbf{c}_k = \sum_{t=1}^T \alpha_{k, t} \mathbf{H}_t$$

### 1.2 Multi-Label Binary Classification
Tool activation logits $z_k \in \mathbb{R}$ are obtained through a linear classification layer with dropout:
$$z_k = \mathbf{w}_k^\top \text{Dropout}(\mathbf{c}_k) + b_k, \quad \mathbf{w}_k \in \mathbb{R}^d, \; b_k \in \mathbb{R}$$

The predicted probability that tool $k$ is triggered is given by the sigmoid function:
$$\hat{p}_k = \sigma(z_k) = \frac{1}{1 + \exp(-z_k)}$$

A tool $k$ is activated if $\hat{p}_k \ge \tau$, where $\tau = 0.5$ is the canonical decision threshold. Multi-tool activation occurs naturally when multiple $\hat{p}_k \ge \tau$.

---

## 2. Linear-Chain CRF Argument Slot Tagging

Let $\mathcal{Y}$ denote the tag inventory consisting of $|\mathcal{Y}| = 26$ BIO slot tags (e.g., `B-message`, `I-message`, `B-url`, `I-url`, `O`, `<PAD>`).

### 2.1 Emission Projections
Token representations $\mathbf{H} \in \mathbb{R}^{T \times d}$ are projected to unnormalized emission scores $\mathbf{E} \in \mathbb{R}^{T \times |\mathcal{Y}|}$:
$$E_{t, y} = \mathbf{w}_y^\top \mathbf{H}_t + b_y, \quad \mathbf{w}_y \in \mathbb{R}^d, \; y \in \mathcal{Y}$$

### 2.2 Hard BIO Transition Masking
A transition matrix $\mathbf{A} \in \mathbb{R}^{|\mathcal{Y}| \times |\mathcal{Y}|}$ models label transition dynamics, alongside start transitions $\mathbf{A}_{\text{start}} \in \mathbb{R}^{|\mathcal{Y}|}$ and end transitions $\mathbf{A}_{\text{end}} \in \mathbb{R}^{|\mathcal{Y}|}$.

To mathematically guarantee that invalid BIO sequences are never decoded, we define the static structural mask $\mathbf{M} \in \{0, -\infty\}^{|\mathcal{Y}| \times |\mathcal{Y}|}$:
$$M_{j, k} = \begin{cases} -\infty & \text{if } j = \text{O} \text{ and } k = \text{I-}X \\ -\infty & \text{if } j \in \{\text{B-}X, \text{I-}X\} \text{ and } k = \text{I-}Y \; (X \neq Y) \\ -\infty & \text{if } j = \text{<PAD>} \text{ and } k \neq \text{<PAD>} \\ 0 & \text{otherwise} \end{cases}$$

Similarly, for start transitions:
$$M_{\text{start}, k} = \begin{cases} -\infty & \text{if } k \in \{\text{I-}X, \text{<PAD>}\} \\ 0 & \text{otherwise} \end{cases}$$

The masked transition matrix $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{M}$ is used in all forward and Viterbi computations.

### 2.3 Sequence Path Score
For an active sequence of length $T$ and label sequence $\mathbf{y} = (y_1, \dots, y_T) \in \mathcal{Y}^T$:
$$s(\mathbf{y}, \mathbf{H}) = \tilde{A}_{\text{start}, y_1} + \sum_{t=1}^T E_{t, y_t} + \sum_{t=1}^{T-1} \tilde{A}_{y_t, y_{t+1}} + \tilde{A}_{\text{end}, y_T}$$

The conditional probability of the target tag sequence $\mathbf{y}$ is:
$$P(\mathbf{y} \mid \mathbf{H}) = \frac{\exp(s(\mathbf{y}, \mathbf{H}))}{Z(\mathbf{H})}$$
where the partition function $Z(\mathbf{H})$ sums over all $|\mathcal{Y}|^T$ possible label sequences:
$$Z(\mathbf{H}) = \sum_{\mathbf{y}' \in \mathcal{Y}^T} \exp(s(\mathbf{y}', \mathbf{H}))$$

### 2.4 Forward Algorithm ($O(T \cdot |\mathcal{Y}|^2)$ Time Complexity)
The log-partition function $\log Z(\mathbf{H})$ is computed efficiently via forward dynamic programming in log-space:
$$\alpha_1(y) = \tilde{A}_{\text{start}, y} + E_{1, y}$$
$$\alpha_t(y) = \bigoplus_{y' \in \mathcal{Y}} \left[ \alpha_{t-1}(y') + \tilde{A}_{y', y} \right] + E_{t, y}, \quad t = 2, \dots, T$$
$$\log Z(\mathbf{H}) = \bigoplus_{y \in \mathcal{Y}} \left[ \alpha_T(y) + \tilde{A}_{\text{end}, y} \right]$$
where $\bigoplus$ denotes the numerically stable `log-sum-exp` operator:
$$\bigoplus_{i=1}^n v_i = \max_i(v_i) + \log \left( \sum_{i=1}^n \exp(v_i - \max_i(v_i)) \right)$$

Complexity:
- Time: $\mathcal{O}(T \cdot |\mathcal{Y}|^2)$ floating-point operations.
- Space: $\mathcal{O}(T \cdot |\mathcal{Y}|)$ forward lattice storage.

### 2.5 Viterbi Decoding ($O(T \cdot |\mathcal{Y}|^2)$ Time Complexity)
During inference, the globally optimal tag sequence $\mathbf{y}^*$ is decoded:
$$\mathbf{y}^* = \arg\max_{\mathbf{y} \in \mathcal{Y}^T} s(\mathbf{y}, \mathbf{H})$$

The Viterbi recurrence:
$$V_1(y) = \tilde{A}_{\text{start}, y} + E_{1, y}$$
$$V_t(y) = \max_{y' \in \mathcal{Y}} \left[ V_{t-1}(y') + \tilde{A}_{y', y} \right] + E_{t, y}, \quad \text{bp}_t(y) = \arg\max_{y' \in \mathcal{Y}} \left[ V_{t-1}(y') + \tilde{A}_{y', y} \right]$$
Best terminal score:
$$V_{\text{final}} = \max_{y \in \mathcal{Y}} \left[ V_T(y) + \tilde{A}_{\text{end}, y} \right], \quad y_T^* = \arg\max_{y \in \mathcal{Y}} \left[ V_T(y) + \tilde{A}_{\text{end}, y} \right]$$
Backtracking $y_{t-1}^* = \text{bp}_t(y_t^*)$ yields $\mathbf{y}^*$ in $\mathcal{O}(T)$ steps.

---

## 3. Loss Normalization Formulation

### 3.1 Slot Loss Length Normalization
The raw negative log-likelihood for sequence $b$ is:
$$\text{NLL}_b = - \log P(\mathbf{y}^{(b)} \mid \mathbf{H}^{(b)}) = \log Z(\mathbf{H}^{(b)}) - s(\mathbf{y}^{(b)}, \mathbf{H}^{(b)})$$

Without length normalization, $\text{NLL}_b \propto T_b$, causing batches with long sentences to dominate parameter updates. To enforce uniform gradient scale across all utterance lengths, the slot loss is normalized per sentence by its active token count:
$$T_b = \sum_{t=1}^{T_{\max}} \text{mask}_{b, t}$$
$$\mathcal{L}_{\text{slot}} = \frac{1}{B} \sum_{b=1}^B \frac{\text{NLL}_b}{T_b}$$

### 3.2 Multi-Label Tool Loss
Multi-label binary cross-entropy with targets $\mathbf{y}^{\text{tool}} \in \{0, 1\}^{B \times K}$:
$$\mathcal{L}_{\text{tool}} = \frac{1}{B \cdot K} \sum_{b=1}^B \sum_{k=1}^K -\left[ y_{b, k}^{\text{tool}} \log \hat{p}_{b, k} + (1 - y_{b, k}^{\text{tool}}) \log (1 - \hat{p}_{b, k}) \right]$$

### 3.3 Joint Objective
$$\mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{tool}} + \lambda_{\text{slot}} \mathcal{L}_{\text{slot}}$$
where $\lambda_{\text{slot}} \in [0.5, 2.0]$ is a weighting hyperparameter tuned strictly on validation loss.

---

## 4. Probability Calibration (ECE)

To calibrate routing confidence, we evaluate Expected Calibration Error (ECE) using FAR sigmoid probabilities $\hat{p}_k = \sigma(z_k)$ rather than raw attention weights.

Given $N$ predictions partitioned into $M=10$ equally spaced confidence bins $B_m = (\frac{m-1}{M}, \frac{m}{M}]$:
$$\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(\hat{y}_i = y_i)$$
$$\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$$
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
An ECE of $0.0$ represents perfect probabilistic calibration.
