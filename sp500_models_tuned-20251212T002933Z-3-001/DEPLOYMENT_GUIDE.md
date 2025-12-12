# 🚀 Deployment Guide for Streamlit Community Cloud

## Step 1: Prepare Your Repository

### 1.1 Initialize Git (if not done)
```bash
git init
git add .gitignore requirements.txt app_v2.py sample_data.csv
git commit -m "Initial commit: S&P 500 Volatility Dashboard"
```

### 1.2 Create a GitHub Repository
1. Go to https://github.com/new
2. Create a new repository (e.g., "sp500-volatility-dashboard")
3. **Do NOT** initialize with README, .gitignore, or license (we already have these)

### 1.3 Push to GitHub
```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

## Step 2: Handle the Model File

⚠️ **Important**: The `.pth` model file is too large for GitHub (>100MB typically).

### Option A: Use Git LFS (Recommended if <2GB)
```bash
# Install Git LFS
git lfs install

# Track .pth files
git lfs track "*.pth"
git add .gitattributes
git add sp500_models_tuned/T_dm64_l2_lr1e4_best.pth
git commit -m "Add model file with Git LFS"
git push
```

### Option B: Use Cloud Storage (Best for production)
1. Upload model to Google Drive, Dropbox, or S3
2. Get a public shareable link
3. Update `app_v2.py` to download the model on startup:

```python
import requests
import os

def download_model():
    model_url = "YOUR_DIRECT_DOWNLOAD_LINK"
    model_path = "sp500_models_tuned/T_dm64_l2_lr1e4_best.pth"
    
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        response = requests.get(model_url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
    return model_path
```

### Option C: Exclude Model (Use smaller version)
If your model is too large, you can:
1. Add `*.pth` to `.gitignore`
2. Train a smaller model or quantize the existing one
3. Or prompt users to upload their own model file

## Step 3: Deploy to Streamlit Community Cloud

### 3.1 Go to Streamlit Cloud
1. Visit https://share.streamlit.io
2. Sign in with your GitHub account

### 3.2 Deploy Your App
1. Click "New app"
2. Select your repository
3. Choose branch: `main`
4. Main file path: `app_v2.py`
5. Click "Deploy"

### 3.3 Configure Settings (Optional)
- **Python version**: 3.11 (recommended)
- **Secrets**: If you have API keys or sensitive data

## Step 4: Update Model Path in Code

Since the model path on Streamlit Cloud will be different, update the default path in `app_v2.py`:

```python
# Change this line in the sidebar:
model_path = st.sidebar.text_input(
    "Model Path",
    value="sp500_models_tuned/T_dm64_l2_lr1e4_best.pth"  # Relative path
)
```

## 🎉 That's It!

Your app will be live at: `https://YOUR_USERNAME-YOUR_REPO_NAME.streamlit.app`

## 🐛 Troubleshooting

### Issue: Model file not found
- Ensure the model file is in the repository or accessible via URL
- Check the path is relative to the app root

### Issue: Out of memory
- Model might be too large for free tier (512MB limit)
- Consider model quantization or use a smaller architecture

### Issue: Slow startup
- Large model files take time to load
- Consider caching with `@st.cache_resource`

### Issue: Dependencies not installing
- Check `requirements.txt` has correct versions
- Ensure all packages are available on PyPI

## 📧 Need Help?

- Streamlit Docs: https://docs.streamlit.io/streamlit-community-cloud
- Streamlit Forum: https://discuss.streamlit.io
- GitHub Issues: Report issues in your repository
