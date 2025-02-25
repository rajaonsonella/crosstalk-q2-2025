# Getting Started

<div style="background: #1e1e2e; color: #cdd6f4; padding: 25px; border-radius: 8px; border-left: 5px solid #cba6f7; font-family: 'JetBrains Mono', monospace; margin: 30px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.2);">
  <h3 style="margin-top: 0; color: #f38ba8; text-align: center; font-family: 'JetBrains Mono', monospace; letter-spacing: 1px;">
    <span style="color: #f9e2af;">⚠️</span> GPU RUNTIME REQUIRED <span style="color: #f9e2af;">⚠️</span>
  </h3>
  
  <div style="background: #313244; padding: 20px; border-radius: 6px; margin: 20px 0; border: 1px solid #45475a;">
    <p style="margin-top: 0; color: #89b4fa; font-weight: bold;">$ runtime.config</p>
    <code style="display: block; color: #a6e3a1; margin: 15px 0; line-height: 1.6;">
      # Follow these steps to configure GPU access:
      
      1. runtime --select "Runtime"  # Menu navigation
      2. runtime --option "Change runtime type"
      3. runtime --set "Hardware accelerator" "GPU"
      4. runtime --apply "Save"
    </code>
    <p style="color: #f5c2e7; font-size: 14px; margin-bottom: 0;">
      <span style="color: #fab387;">></span> GPU acceleration will improve performance by ~8-12x
    </p>
  </div>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 30px 0;">
  <div style="background: #1e1e2e; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.2);">
    <div style="background: #313244; padding: 15px; border-bottom: 1px solid #45475a; display: flex; align-items: center;">
      <div style="width: 12px; height: 12px; background-color: #f38ba8; border-radius: 50%; margin-right: 8px;"></div>
      <div style="width: 12px; height: 12px; background-color: #f9e2af; border-radius: 50%; margin-right: 8px;"></div>
      <div style="width: 12px; height: 12px; background-color: #a6e3a1; border-radius: 50%; margin-right: 15px;"></div>
      <span style="color: #cdd6f4; font-family: 'JetBrains Mono', monospace; font-size: 14px;">EDA.ipynb</span>
    </div>
    <div style="padding: 20px; text-align: center;">
      <p style="color: #89b4fa; font-size: 15px; margin-top: 0; font-family: 'JetBrains Mono', monospace;">Exploratory Data Analysis</p>
      <a href="https://colab.research.google.com/github/cottascience/crosstalk-q1-2025/blob/main/EDA.ipynb" style="display: inline-block; text-decoration: none; margin-top: 15px;">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
      </a>
    </div>
  </div>

  <div style="background: #1e1e2e; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.2);">
    <div style="background: #313244; padding: 15px; border-bottom: 1px solid #45475a; display: flex; align-items: center;">
      <div style="width: 12px; height: 12px; background-color: #f38ba8; border-radius: 50%; margin-right: 8px;"></div>
      <div style="width: 12px; height: 12px; background-color: #f9e2af; border-radius: 50%; margin-right: 8px;"></div>
      <div style="width: 12px; height: 12px; background-color: #a6e3a1; border-radius: 50%; margin-right: 15px;"></div>
      <span style="color: #cdd6f4; font-family: 'JetBrains Mono', monospace; font-size: 14px;">notebook.ipynb</span>
    </div>
    <div style="padding: 20px; text-align: center;">
      <p style="color: #89b4fa; font-size: 15px; margin-top: 0; font-family: 'JetBrains Mono', monospace;">Training Notebook</p>
      <a href="https://colab.research.google.com/github/cottascience/crosstalk-q1-2025/blob/main/notebook.ipynb" style="display: inline-block; text-decoration: none; margin-top: 15px;">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
      </a>
    </div>
  </div>
</div>
