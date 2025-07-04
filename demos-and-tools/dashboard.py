import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path
import time
import os
from PIL import Image

st.set_page_config(page_title="🧰 Semantic Uncertainty Diagnostic Suite", layout="wide", page_icon="🧠")

st.title("🧰 Semantic Uncertainty Diagnostic Suite")
st.markdown("**Purpose:** Profile cognition under strain, not rank performance")
st.markdown("**Protocol:** 5-step model-agnostic evaluation")
st.markdown("---")

# Data loading functions
@st.cache_data(show_spinner=False)
def load_diagnostic_data():
    """Load all diagnostic data from the outputs directory"""
    data = {}
    
    # Check both old and new paths for backwards compatibility
    output_dirs = [
        Path("data-and-results/diagnostic_outputs"),
        Path("diagnostic_outputs"),  # Legacy path
        Path("data-and-results/evaluation_outputs"),
        Path("results")  # Legacy path
    ]
    
    # Load basic results if available
    for results_file in ["results.csv", "all_results.csv"]:
        if os.path.exists(results_file):
            data['basic_results'] = pd.read_csv(results_file)
            break
    
    # Load diagnostic suite outputs from any available directory
    for output_dir in output_dirs:
        if not output_dir.exists():
            continue
        # Normalized clusters
        clusters_file = output_dir / "normalized_prompt_clusters.json"
        if clusters_file.exists():
            with open(clusters_file) as f:
                data['clusters'] = json.load(f)
        
        # Calibration data
        calib_file = output_dir / "calibrated_delta_hbar_table.csv"
        if calib_file.exists():
            data['calibration'] = pd.read_csv(calib_file)
        
        # Robustness curves
        robust_file = output_dir / "robustness_curves.json"
        if robust_file.exists():
            with open(robust_file) as f:
                data['robustness'] = json.load(f)
        
        # Sensitivity map
        sens_file = output_dir / "collapse_sensitivity_map.csv"
        if sens_file.exists():
            data['sensitivity'] = pd.read_csv(sens_file)
        
        # Summary report
        summary_file = output_dir / "diagnostic_summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                data['summary'] = json.load(f)
        
        # Load collapse profiles
        data['profiles'] = {}
        for profile_file in output_dir.glob("collapse_profile_*.json"):
            model_name = profile_file.stem.replace("collapse_profile_", "")
            with open(profile_file) as f:
                data['profiles'][model_name] = json.load(f)
        
        # Load heatmaps
        data['heatmaps'] = {}
        for heatmap_file in output_dir.glob("heatmap_projection_*.png"):
            model_name = heatmap_file.stem.replace("heatmap_projection_", "")
            data['heatmaps'][model_name] = str(heatmap_file)
    
    return data

# Load data
data = load_diagnostic_data()

# Sidebar with refresh and info
with st.sidebar:
    st.header("🔄 Controls")
    if st.button("🔄 Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    st.header("💾 Export Options")
    if 'calibration' in data or 'basic_results' in data:
        if st.button("📁 Export All Results", type="secondary"):
            # Create organized export
            export_dir = Path("data-and-results/dashboard_exports")
            export_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            export_path = export_dir / f"semantic_analysis_{timestamp}"
            export_path.mkdir(exist_ok=True)
            
            if 'calibration' in data:
                data['calibration'].to_csv(export_path / "calibration_data.csv", index=False)
            if 'basic_results' in data:
                data['basic_results'].to_csv(export_path / "evaluation_results.csv", index=False)
            
            st.success(f"✅ Exported to: {export_path}")
            st.info("📦 Contains: CSV files ready for analysis")
    else:
        st.info("💡 Run evaluations with --save flag to enable exports")
    
    st.header("📊 Data Status")
    if 'calibration' in data:
        st.success(f"✅ Diagnostic Suite Complete")
        st.info(f"📈 {len(data['calibration'])} evaluations")
        st.info(f"🤖 {data['calibration']['model'].nunique()} models")
        st.info(f"🎯 {data['calibration']['category'].nunique()} categories")
    elif 'basic_results' in data:
        st.success("✅ Basic Demo Complete")
        st.info(f"📈 {len(data['basic_results'])} evaluations")
    else:
        st.warning("⏳ No results yet")
        st.info("🔧 Terminal Mode:")
        st.code("python evaluation-frameworks/diagnostic_suite_simplified.py")
        st.info("💾 Save Mode:")
        st.code("python evaluation-frameworks/diagnostic_suite_simplified.py --save")
    
    st.header("🧠 About ℏₛ")
    st.markdown("""
    **Semantic Uncertainty Principle:**
    
    ℏₛ(C) = √(Δμ × Δσ)
    
    - **Δμ**: Semantic precision
    - **Δσ**: Semantic flexibility  
    - **ℏₛ**: Semantic uncertainty
    
    This is **not** a leaderboard score.
    It's a **stress tensor on meaning**.
    """)

# Main content tabs
if 'calibration' in data:
    # Full diagnostic suite view
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "🔁 Normalization", "🧩 Calibration", 
        "🧠 Probing", "📈 Heatmaps", "🔬 Profiles"
    ])
    
    with tab1:
        st.header("📊 Diagnostic Overview")
        
        if 'summary' in data:
            summary = data['summary']
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Models Analyzed", summary['models_analyzed'])
            with col2:
                st.metric("Total Evaluations", summary['total_evaluations'])
            with col3:
                if 'key_findings' in summary:
                    st.metric("Avg Perturbation Sensitivity", 
                             f"{summary['key_findings']['average_perturbation_sensitivity']:.3f}")
            with col4:
                if 'key_findings' in summary:
                    st.metric("Tier 3 Collapse Rate", 
                             f"{summary['key_findings']['tier_3_average_collapse']:.3f}")
        
        # Display comparative analysis image
        comp_analysis_path = Path("diagnostic_outputs/comparative_analysis.png")
        if comp_analysis_path.exists():
            st.subheader("🎯 Comparative Analysis")
            st.image(str(comp_analysis_path), use_column_width=True)
        
        # Results table
        st.subheader("📋 Detailed Results")
        df = data['calibration']
        
        # Add color coding based on collapse risk
        def color_collapse_risk(val):
            if val:
                return 'background-color: #ffcccc'  # Light red
            else:
                return 'background-color: #ccffcc'  # Light green
        
        styled_df = df.style.applymap(color_collapse_risk, subset=['collapse_risk'])
        st.dataframe(styled_df, use_container_width=True)
    
    with tab2:
        st.header("🔁 Step 1: Prompt Normalization")
        st.markdown("**Goal:** Normalize prompt length, ensure >95% semantic similarity, cluster into identity classes")
        
        if 'clusters' in data:
            clusters = data['clusters']
            
            # Cluster summary
            st.subheader("📊 Cluster Summary")
            cluster_data = []
            for category, cluster_info in clusters.items():
                cluster_data.append({
                    'Category': category,
                    'Tier': cluster_info['tier'],
                    'Canonical Prompt': cluster_info['canonical_prompt'],
                    'Paraphrases': len(cluster_info['paraphrases']),
                    'Mean Similarity': f"{cluster_info['mean_similarity']:.3f}"
                })
            
            cluster_df = pd.DataFrame(cluster_data)
            st.dataframe(cluster_df, use_container_width=True)
            
            # Detailed view
            st.subheader("🔍 Detailed Cluster Analysis")
            selected_category = st.selectbox("Select Category:", list(clusters.keys()))
            
            if selected_category:
                cluster = clusters[selected_category]
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write("**Canonical Prompt:**")
                    st.code(cluster['canonical_prompt'])
                    
                    st.write("**Valid Paraphrases:**")
                    for i, paraphrase in enumerate(cluster['paraphrases'], 1):
                        st.write(f"{i}. {paraphrase}")
                
                with col2:
                    st.metric("Tier", cluster['tier'])
                    st.metric("Mean Similarity", f"{cluster['mean_similarity']:.3f}")
                    st.write("**Token Counts by Model:**")
                    for model, count in cluster['token_counts'].items():
                        st.write(f"• {model}: {count}")
    
    with tab3:
        st.header("🧩 Step 2: Calibration Set Construction")
        st.markdown("**Goal:** Build tiered semantic stress set, compute Δℏₛ(C, model)")
        
        df = data['calibration']
        
        # Tier analysis
        st.subheader("📊 Performance by Tier")
        tier_stats = df.groupby(['tier', 'model'])['hbar_s'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        for tier in sorted(df['tier'].unique()):
            tier_data = tier_stats[tier_stats['tier'] == tier]
            ax.bar([f"{model}\nT{tier}" for model in tier_data['model']], 
                   tier_data['hbar_s'], 
                   alpha=0.7, 
                   label=f'Tier {tier}')
        
        ax.set_ylabel('Mean ℏₛ')
        ax.set_title('Mean ℏₛ by Model and Tier')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Delta hbar_s analysis
        st.subheader("📈 Δℏₛ Analysis")
        st.markdown("Δℏₛ(C, model) = ℏₛ(model)(C) - ℏ̄ₛ(C)")
        
        delta_stats = df.groupby('model')['delta_hbar_s'].agg(['mean', 'std']).reset_index()
        delta_stats.columns = ['Model', 'Mean Δℏₛ', 'Std Δℏₛ']
        st.dataframe(delta_stats, use_container_width=True)
    
    with tab4:
        st.header("🧠 Step 3: Information-Aligned Probing")
        st.markdown("**Goal:** Compute ℏₛ slopes across tiers, analyze perturbation sensitivity")
        
        if 'robustness' in data and 'sensitivity' in data:
            robustness = data['robustness']
            sensitivity_df = data['sensitivity']
            
            # Tier slopes
            st.subheader("📉 ℏₛ Slopes Across Tiers")
            slope_data = []
            for key, value in robustness.items():
                if '_tier_slope' in key:
                    model = key.replace('_tier_slope', '')
                    slope_data.append({'Model': model, 'Tier Slope': value})
            
            if slope_data:
                slope_df = pd.DataFrame(slope_data)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(slope_df['Model'], slope_df['Tier Slope'], alpha=0.7)
                ax.set_ylabel('ℏₛ Slope (Tier 1 → 3)')
                ax.set_title('Brittle Generalization Detection')
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                
                # Color bars based on slope
                for bar, slope in zip(bars, slope_df['Tier Slope']):
                    if slope < -0.1:
                        bar.set_color('red')  # Degrading
                    elif slope > 0.1:
                        bar.set_color('green')  # Improving
                    else:
                        bar.set_color('blue')  # Stable
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Perturbation sensitivity
            st.subheader("🌊 Perturbation Sensitivity")
            sens_data = []
            for key, value in robustness.items():
                if '_perturbation_sensitivity' in key:
                    model = key.replace('_perturbation_sensitivity', '')
                    sens_data.append({'Model': model, 'Sensitivity': value})
            
            if sens_data:
                sens_df = pd.DataFrame(sens_data)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(sens_df['Model'], sens_df['Sensitivity'], alpha=0.7)
                ax.set_ylabel('Perturbation Sensitivity')
                ax.set_title('Sensitivity to Rephrasing')
                ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='High Sensitivity Threshold')
                ax.legend()
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    with tab5:
        st.header("📈 Step 4: Dimension-Reduced Heatmaps")
        st.markdown("**Goal:** Project semantic terrain into 2D space, generate contour plots")
        
        if 'heatmaps' in data:
            heatmaps = data['heatmaps']
            
            st.subheader("🗺️ Semantic Terrain Maps")
            
            # Model selection
            selected_models = st.multiselect(
                "Select models to view:",
                list(heatmaps.keys()),
                default=list(heatmaps.keys())[:4]  # Show first 4 by default
            )
            
            # Display heatmaps in grid
            cols_per_row = 2
            for i in range(0, len(selected_models), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, model in enumerate(selected_models[i:i+cols_per_row]):
                    with cols[j]:
                        st.subheader(f"🤖 {model}")
                        if model in heatmaps:
                            try:
                                img = Image.open(heatmaps[model])
                                st.image(img, use_column_width=True)
                            except Exception as e:
                                st.error(f"Could not load heatmap: {e}")
        else:
            st.info("No heatmaps generated yet. Run the diagnostic suite first.")
    
    with tab6:
        st.header("🔬 Step 5: Semantic Collapse Profiles")
        st.markdown("**Goal:** Generate failure fingerprints, analyze where/when/how models fail")
        
        if 'profiles' in data:
            profiles = data['profiles']
            
            # Model selection
            selected_model = st.selectbox("Select Model for Detailed Profile:", list(profiles.keys()))
            
            if selected_model and selected_model in profiles:
                profile = profiles[selected_model]
                
                # Key metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Perturbation Sensitivity", f"{profile['perturbation_sensitivity']:.3f}")
                with col2:
                    st.metric("ℏₛ Drop Sharpness", f"{profile['hbar_s_drop_sharpness']:.3f}")
                with col3:
                    terrain_x, terrain_y = profile['semantic_terrain_coords']
                    st.metric("Semantic Terrain", f"({terrain_x:.2f}, {terrain_y:.2f})")
                
                # Tier collapse rates
                st.subheader("📊 Collapse Rates by Tier")
                tier_rates = profile['tier_collapse_rates']
                
                fig, ax = plt.subplots(figsize=(8, 5))
                tiers = list(tier_rates.keys())
                rates = list(tier_rates.values())
                bars = ax.bar([f"Tier {t}" for t in tiers], rates, alpha=0.7)
                
                # Color code by collapse thresholds
                thresholds = {1: 0.45, 2: 0.40, 3: 0.35}
                for bar, tier, rate in zip(bars, tiers, rates):
                    if rate > 0.5:
                        bar.set_color('red')
                    elif rate > 0.3:
                        bar.set_color('orange')
                    else:
                        bar.set_color('green')
                
                ax.set_ylabel('Collapse Rate')
                ax.set_title(f'Collapse Profile: {selected_model}')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Category vulnerability
                st.subheader("🎯 Category Vulnerability")
                cat_vuln = profile['category_vulnerability']
                vuln_df = pd.DataFrame(list(cat_vuln.items()), columns=['Category', 'Collapse Rate'])
                vuln_df = vuln_df.sort_values('Collapse Rate', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(vuln_df['Category'], vuln_df['Collapse Rate'], alpha=0.7)
                
                # Color code
                for bar, rate in zip(bars, vuln_df['Collapse Rate']):
                    if rate > 0.5:
                        bar.set_color('red')
                    elif rate > 0.3:
                        bar.set_color('orange')
                    else:
                        bar.set_color('green')
                
                ax.set_ylabel('Collapse Rate')
                ax.set_title('Vulnerability by Category')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Failure patterns
                st.subheader("⚠️ Failure Patterns")
                if profile['failure_patterns']:
                    for pattern in profile['failure_patterns']:
                        st.warning(f"• {pattern}")
                else:
                    st.success("✅ No significant failure patterns detected")

elif 'basic_results' in data:
    # Basic demo view
    st.header("🧪 Basic Demo Results")
    
    df = data['basic_results']
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Evaluations", len(df))
    with col2:
        st.metric("Models Tested", df['model'].nunique())
    with col3:
        st.metric("Average ℏₛ", f"{df['hbar_s'].mean():.3f}")
    with col4:
        collapse_rate = df['collapse_risk'].mean()
        st.metric("Collapse Rate", f"{collapse_rate:.1%}")
    
    # Results table
    st.subheader("📋 Results Table")
    st.dataframe(df, use_container_width=True)
    
    # Basic visualizations
    st.subheader("📊 Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots()
        model_stats = df.groupby('model')['hbar_s'].mean()
        model_stats.plot(kind='bar', ax=ax, alpha=0.7)
        ax.set_title('Average ℏₛ per Model')
        ax.set_ylabel('ℏₛ')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        fig, ax = plt.subplots()
        collapse_stats = df.groupby('model')['collapse_risk'].mean()
        collapse_stats.plot(kind='bar', ax=ax, alpha=0.7, color='coral')
        ax.set_title('Collapse Rate per Model')
        ax.set_ylabel('Collapse Rate')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

else:
    # No data available
    st.header("⏳ No Results Available")
    st.info("Run one of the following to generate results:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 Basic Demo")
        st.code("python semantic-uncertainty-runtime/quick_demo.py")
        st.markdown("Runs a simple evaluation with mock responses")
    
    with col2:
        st.subheader("🧰 Full Diagnostic Suite")
        st.code("python semantic-uncertainty-runtime/diagnostic_suite_simplified.py")
        st.markdown("Runs the complete 5-step model-agnostic evaluation protocol")

# Footer
st.markdown("---")
st.markdown("🧠 **Remember:** ℏₛ(C) is not a leaderboard score. It's a stress tensor on meaning.")
st.markdown("🔬 **Purpose:** Profile cognition under strain, not rank performance") 