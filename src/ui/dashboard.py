import streamlit as st
import pandas as pd
import plotly.express as px
from src.engine.pipeline import A2MLPipeline
import io

def render_dashboard():
    st.markdown("<h1>A²ML: The Autonomous Intelligence Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1rem; color: #94a3b8; margin-bottom: 2rem;'>An intelligent, self-learning pipeline that autonomously constructs, optimizes, and evaluates predictive models.</p>", unsafe_allow_html=True)
    
    st.markdown("<h3>1. Data Injection Layer</h3>", unsafe_allow_html=True)
    
    col_upload, col_settings = st.columns([1, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader("Drop your CSV Knowledge Base here", type=["csv"])
        
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        with col_settings:
            target_col = st.selectbox("Define Target Field (Leave auto to let engine infer)", ["-- Auto Detect --"] + list(df.columns))
            feature_strategy = st.selectbox("Feature Engineering Strategy", ["auto", "pca", "mutual_info", "none"])
            t_col = None if target_col == "-- Auto Detect --" else target_col
        
        if st.button("Run Autonomous Pipeline", type="primary"):
            with st.spinner("A²ML is analyzing dataset, selecting models, and optimizing hyperparameters..."):
                try:
                    # Reset pointer
                    uploaded_file.seek(0)
                    pipeline = A2MLPipeline(uploaded_file)
                    results = pipeline.run_pipeline(
                        target_column=t_col,
                        feature_opt_strategy=feature_strategy
                    )
                    
                    st.success("Pipeline executed successfully!")
                    
                    display_results(results)
                except Exception as e:
                    st.error(f"Error during execution: {e}")

def display_results(results):
    st.markdown("<br><hr style='border-color: rgba(255,255,255,0.1);'><br>", unsafe_allow_html=True)
    st.markdown("<h3>2. Meta-Learning Intelligence Report</h3>", unsafe_allow_html=True)
    
    rep = results["dataset_report"]
    
    # Use metrics for a card-like effect
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Problem Topology", results["problem_type"].capitalize())
    with col2:
        st.metric("Predicted Target", results["target"])
    with col3:
        st.metric("Analyzed Vectors", f"{rep['num_rows']:,}")
    with col4:
        st.metric("Feature Dimensions", f"{rep['num_columns']}")
        
    st.markdown(f"<p style='color: #cbd5e1;'>Data Complexity Index: <b>{rep['data_complexity_score']}</b></p>", unsafe_allow_html=True)

    if results["recommended_from_memory"]:
         st.markdown(f"<div style='padding: 1rem; border-radius: 8px; background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); color: #c7d2fe; margin-bottom: 2rem;'>🧠 <b>A²ML Memory Linkage</b>: Based on exact historical topography matching, the engine recommended prioritizing <b>{results['recommended_from_memory']}</b>.</div>", unsafe_allow_html=True)
         
    st.markdown("<h3>3. Neural Benchmark Matrix</h3>", unsafe_allow_html=True)
    bench = results["benchmark_results"]
    st.dataframe(bench.style.highlight_max(axis=0, color="rgba(168, 85, 247, 0.4)"), use_container_width=True)
    
    st.markdown(f"<div style='padding: 1.5rem; text-align: center; border-radius: 12px; background: linear-gradient(90deg, rgba(99,102,241,0.2) 0%, rgba(168,85,247,0.2) 100%); border: 1px solid rgba(168,85,247,0.5); font-size: 1.2rem; font-weight: 600; color: white;'>🏆 Top Performing Network Synthesized: {results['best_model_name']}</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_plot1, col_plot2 = st.columns([1, 1])
    
    with col_plot1:
        # Plot benchmark
        if results["problem_type"] == "classification":
             fig = px.bar(bench, x='Model', y='F1 Score', title="F1 Score Performance", color='F1 Score', color_continuous_scale="Purples")
        elif results["problem_type"] in ["regression", "time_series"]:
             fig = px.bar(bench, x='Model', y='RMSE', title="RMSE Depletion (Lower is Better)", color='RMSE', color_continuous_scale="Purples_r")
             
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig, use_container_width=True)
         
    with col_plot2:
        st.markdown("<h4>Explainable AI (XAI) Matrix</h4>", unsafe_allow_html=True)
        imp = results["explanation"]
        pdp = results.get("pdp", None)
        
        if imp:
            # SHAP Bar Chart
            df_imp = pd.DataFrame(list(imp.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=True)
            fig2 = px.bar(df_imp, x='Importance', y='Feature', orientation='h', title=f"SHAP Influence map: {results['best_model_name']}", color='Importance', color_continuous_scale="Teal")
            fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white", height=350)
            st.plotly_chart(fig2, use_container_width=True)
            
            # PDP Line Chart
            if pdp:
                st.markdown(f"**Partial Dependence:** `{pdp['feature']}`", unsafe_allow_html=True)
                df_pdp = pd.DataFrame({pdp['feature']: pdp['values'], "Dependence": pdp['dependence']})
                fig3 = px.line(df_pdp, x=pdp['feature'], y='Dependence', markers=True)
                fig3.update_traces(line_color="#a855f7")
                fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white", height=300)
                st.plotly_chart(fig3, use_container_width=True)
                
        else:
            st.markdown("<p style='color: #94a3b8;'>Feature importance extraction bypass engaged for this matrix configuration.</p>", unsafe_allow_html=True)
