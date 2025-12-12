# 📝 NEXT STEPS: Deploy Your Streamlit App

## ✅ What We've Done

1. ✓ Created `.gitignore` to exclude unnecessary files
2. ✓ Initialized Git repository
3. ✓ Committed essential files (app, requirements, sample data, documentation)
4. ✓ Created deployment guide

## 🚀 What You Need to Do Next

### Step 1: Create a GitHub Repository

1. **Go to GitHub**: https://github.com/new
2. **Repository name**: `sp500-volatility-dashboard` (or your preferred name)
3. **Description**: "S&P 500 Volatility Forecasting Dashboard with Transformer Model"
4. **Visibility**: Public (required for free Streamlit deployment)
5. **Important**: Do NOT check any boxes (no README, .gitignore, or license)
6. Click "Create repository"

### Step 2: Push Your Code to GitHub

Copy and run these commands in your terminal (replace with your GitHub username and repo name):

```bash
git remote add origin https://github.com/YOUR_USERNAME/sp500-volatility-dashboard.git
git branch -M main
git push -u origin main
```

Example:
```bash
git remote add origin https://github.com/johndoe/sp500-volatility-dashboard.git
git branch -M main
git push -u origin main
```

### Step 3: Handle the Model File

⚠️ **IMPORTANT**: The `.pth` model file (156MB+) is too large for GitHub.

**Choose ONE option:**

#### Option A: Git LFS (If model < 2GB)
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add sp500_models_tuned/T_dm64_l2_lr1e4_best.pth
git commit -m "Add model file with Git LFS"
git push
```

#### Option B: Google Drive (Recommended - Simple & Free)
1. Upload `sp500_models_tuned/T_dm64_l2_lr1e4_best.pth` to Google Drive
2. Right-click → Share → Get link → "Anyone with the link"
3. Copy the file ID from the URL (the long string after `/d/`)
4. Update the model loading code (see DEPLOYMENT_GUIDE.md)

#### Option C: Prompt users to upload their own model
- Keep model excluded from git
- Users upload the .pth file through the Streamlit interface

### Step 4: Deploy to Streamlit Community Cloud

1. **Go to**: https://share.streamlit.io
2. **Sign in** with your GitHub account
3. Click **"New app"**
4. **Select**:
   - Repository: `YOUR_USERNAME/sp500-volatility-dashboard`
   - Branch: `main`
   - Main file path: `app_v2.py`
5. Click **"Deploy"**

### Step 5: Update Model Path (After Deployment)

Once deployed, update the model path in the Streamlit sidebar to point to the correct location:
- If using Git LFS: `sp500_models_tuned/T_dm64_l2_lr1e4_best.pth`
- If using cloud storage: Enter the download URL

## 📋 Quick Command Reference

```bash
# 1. Add remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# 2. Rename branch to main
git branch -M main

# 3. Push to GitHub
git push -u origin main

# 4. (Optional) Add model with Git LFS
git lfs install
git lfs track "*.pth"
git add .gitattributes sp500_models_tuned/*.pth
git commit -m "Add model with Git LFS"
git push
```

## 🎯 Your App Will Be Live At

```
https://YOUR_USERNAME-sp500-volatility-dashboard.streamlit.app
```

## 📚 Additional Resources

- **Streamlit Docs**: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app
- **Git LFS**: https://git-lfs.github.com/
- **Troubleshooting**: See `DEPLOYMENT_GUIDE.md`

## ❓ Need Help?

If you encounter issues:
1. Check `DEPLOYMENT_GUIDE.md` for detailed troubleshooting
2. Visit Streamlit Community Forum: https://discuss.streamlit.io
3. Check GitHub repository settings (must be public for free tier)

---

## 🎉 Summary

Your files are ready to deploy! Just:
1. Create GitHub repo
2. Push code with `git push`
3. Deploy on share.streamlit.io
4. Share your live app URL!

**Estimated time**: 5-10 minutes ⏱️
