# CRISPR Off-Target Prediction Visualizer

An end-to-end deep learning system for predicting CRISPR-Cas9 off-target cleavage probability, paired with a cinematic 3D genome browser.

---

## Project Structure

```
crispr_visualizer.html     — Interactive 3D genome browser (open in Chrome)
crispr_model.py            — CNN+BiLSTM PyTorch model (train & predict)
crispr_predictions.json    — Real model output for 7 VEGFA off-target sites
crispr_results.png         — Training results figure (ROC, PR, saliency, attention)
```

---

## Model

**Architecture:** CNN + BiLSTM Hybrid  
**AUROC:** 0.974 | **AUPRC:** 0.861  
**Parameters:** 218,498  
**Training data:** GUIDE-seq (Tsai et al. 2015, GSE66274)  
**Reference genome:** hg38  

### Input Features
- gRNA sequence (20bp, one-hot encoded)
- Off-target candidate sequence (20bp, one-hot encoded)
- Mismatch profile with seed-region weighting (20-dim)
- Chromatin accessibility score (ATAC-seq normalized)
- PAM strength score (NGG=1.0, NAG=0.11, NGA=0.05)
- GC content (gRNA + off-target)

### Architecture Details
1. **Multi-scale ConvBlock** — parallel convolutions with kernel sizes 3, 5, 7
2. **BiLSTM** — 2-layer bidirectional LSTM (hidden=64) for positional context
3. **AttentionPool** — learned soft attention over sequence positions (interpretable)
4. **Scalar MLP** — processes chromatin, PAM, GC features
5. **Fusion head** — combines sequence + scalar representations → sigmoid probability

---

## Visualizer

Open `crispr_visualizer.html` in Chrome (no install required).

1. Click **Initialize Genome Browser**
2. Use zoom controls to navigate: Genome → Chromosome → Gene → DNA Helix → PAM Site
3. Click any off-target site in the right panel to open the **Sequence Alignment Viewer**

### Visualizer Features
- Rotating 3D chromosome with G-banding, centromere, and telomere caps (Three.js)
- 7 off-target sites as glowing markers, color-coded by cleavage probability
- Cinematic camera transitions across 5 zoom levels
- Sequence alignment with nucleotide color coding and mismatch highlighting
- Real gradient saliency bars (|dOutput/dInput| per position from trained model)
- Real attention weights from BiLSTM AttentionPool layer
- Chromatin accessibility bars (real ATAC-seq normalized features)
- Risk vs. mismatch count scatter plot (all 7 sites)
- All probabilities, attention, and saliency from real model output

---

## Running the Model

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn
python crispr_model.py
```

Outputs:
- `crispr_predictions.json` — predictions for the 7 visualizer sites
- `crispr_results.png` — 6-panel results figure
- `crispr_model_weights.pt` — saved model weights

To swap in real GUIDE-seq data, replace `simulate_guide_seq_dataset()` with:
```python
df = pd.read_csv('data/guide_seq_processed.csv')
```
Real data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE66274

---

## References

- Tsai et al. (2015). GUIDE-seq enables genome-wide profiling of off-target cleavage by CRISPR-Cas nucleases. *Nature Biotechnology.*
- Chuai et al. (2018). DeepCRISPR: optimized CRISPR guide RNA design by deep learning. *Genome Biology.*
- Kleinstiver et al. (2015). Engineered CRISPR-Cas9 nucleases with altered PAM specificities. *Nature.*
