# Streamlit Cloud Deployment Guide

## 🌐 Deploy Your App to Streamlit Cloud (FREE!)

Your students can access the app online without installing anything!

### Step-by-Step Deployment

#### 1. Go to Streamlit Cloud

Visit: **https://share.streamlit.io/**

#### 2. Sign In with GitHub

- Click **"Sign in with GitHub"**
- Authorize Streamlit to access your GitHub account
- You may need to install the Streamlit app for your GitHub account

#### 3. Deploy New App

- Click **"New app"** button (or "Create app")
- You'll see a form with three fields:

**Repository**: `nvdewegh/spatiotemporal-analysis-tool`

**Branch**: `main`

**Main file path**: `streamlit_deploy/streamlit_visualization.py`

#### 4. Advanced Settings (Optional but Recommended)

Click "Advanced settings" and set:

**Python version**: `3.10` (or higher)

**Requirements file**: `requirements.txt` (auto-detected)

#### 5. Deploy!

- Click **"Deploy!"** button
- Wait 2-5 minutes for deployment
- Your app will be live at a URL like:
  `https://spatiotemporal-analysis-tool.streamlit.app`

---

## 🎓 Sharing with Students

Once deployed, share this link with your students:

**Your app URL**: `https://[your-app-name].streamlit.app`

They can:
- ✅ Use it directly in their browser
- ✅ Upload their own trajectory data
- ✅ Run all analyses without installing anything
- ✅ Download results as CSV files

---

## ⚙️ Managing Your App

### View App Dashboard

Go to https://share.streamlit.io/

You can:
- **View logs**: See errors and activity
- **Reboot app**: Restart if it crashes
- **Update settings**: Change Python version or requirements
- **Delete app**: Remove deployment

### Auto-Updates

Whenever you push changes to GitHub, Streamlit Cloud will:
- Automatically detect the changes
- Redeploy the app
- Keep your students using the latest version!

---

## 📊 Resource Limits (Free Tier)

Streamlit Community Cloud is free but has limits:

- **1 GB RAM** per app
- **1 CPU** per app
- **Unlimited** public apps
- Apps sleep after **7 days** of inactivity (auto-wake on visit)

**For this tool**: The free tier is sufficient for educational use with typical trajectory datasets (<1000 trajectories).

---

## 🔒 Making Repository Public (Required)

Streamlit Cloud requires **public** GitHub repositories for free hosting.

To check/make your repo public:

1. Go to: https://github.com/nvdewegh/spatiotemporal-analysis-tool
2. Click **Settings** (top right)
3. Scroll to bottom → **Danger Zone**
4. Click **Change visibility** → **Make public**

---

## 🎯 Two Options for Students

### Option A: Use Cloud Version (Recommended for most)
**Pros**: No installation, instant access
**Cons**: Requires internet, shares server resources
**Best for**: Quick analyses, demonstrations, students without Python experience

### Option B: Download and Run Locally
**Pros**: Full control, works offline, faster for large datasets
**Cons**: Requires Python installation
**Best for**: Heavy computational work, students learning Python

You can offer both options!

---

## 📝 Example Student Instructions

### For Cloud Version:

```
1. Go to: https://[your-app-url].streamlit.app
2. Upload your trajectory data (CSV or Excel)
3. Follow the 7-step workflow
4. Download your results
```

### For Local Version:

```
1. Download from: https://github.com/nvdewegh/spatiotemporal-analysis-tool
2. Follow INSTALLATION.md
3. Run locally on your computer
```

---

## 🐛 Troubleshooting Deployment

### App won't deploy

**Check**:
- Repository is public ✓
- `requirements.txt` is in root directory ✓
- `streamlit_visualization.py` path is correct ✓
- All package versions are compatible ✓

### App crashes or shows errors

**Solutions**:
- Check **Manage app → Logs** for error messages
- Verify all imports are in `requirements.txt`
- Test locally first before deploying
- Consider memory limits (1 GB max)

### App is slow

**Solutions**:
- Free tier has limited resources
- Consider data preprocessing to reduce size
- Optimize code for memory efficiency
- For large datasets, recommend local installation

---

## 🎉 Success Checklist

- [ ] GitHub repository is public
- [ ] Code pushed to GitHub successfully
- [ ] Signed in to share.streamlit.io
- [ ] App deployed successfully
- [ ] App URL works in browser
- [ ] Tested uploading sample data
- [ ] All 7 steps work correctly
- [ ] Shared URL with students

---

## 📧 Sample Email to Students

```
Subject: Trajectory Clustering Analysis Tool - Now Available!

Dear Students,

I'm excited to share our new Trajectory Clustering Analysis Tool!

🌐 ONLINE VERSION (No Installation):
https://[your-app-url].streamlit.app

Simply upload your trajectory data and follow the 7-step workflow.

💻 LOCAL VERSION (For advanced users):
https://github.com/nvdewegh/spatiotemporal-analysis-tool
Download and follow INSTALLATION.md

Questions? Check the Documentation tab (Step 7) in the app!

Best regards,
[Your Name]
```

---

**Your app is now accessible worldwide!** 🚀
