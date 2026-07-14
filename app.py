import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
import shap
import streamlit.components.v1 as components


@st.cache_resource # Demande à Streamlit de ne charger ces fichiers qu'une seule fois
def charger_modele_logistic():
    modele = joblib.load('models/pipeline_logistique.pkl')
    colonnes = joblib.load('models/colonnes_entrainement.pkl')
    return modele, colonnes
def charger_modele_tree():
    modele = joblib.load('modele/decisiontree.pkl')
    return modele

def reinit():
    st.session_state.disprf ="Menu RF Model"
    st.session_state.disp ="Menu Logistic Model"
    st.session_state.dispositif = "Welcome"



def afficher_pfi():
    df_pfi = joblib.load('models/pfi_results.pkl')
    df_pfi = df_pfi[df_pfi['Importance'] > 0]
    df_pfi = df_pfi.sort_values(by='Importance', ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars = ax.barh(df_pfi['Feature'], df_pfi['Importance'], color='#1abc9c', height=0.6)
    for bar in bars:
        valeur_brute = bar.get_width()
        pourcentage = valeur_brute * 100 # On transforme 0.045 en 4.5%
        ax.text(valeur_brute + 0.001, bar.get_y() + bar.get_height()/2, 
                f'-{pourcentage:.2f}%', 
                va='center', ha='left', fontsize=10, fontweight='bold', color='black')
    ax.set_xlabel('Accuracy Loss (%)', fontsize=12, fontweight='bold')
    ax.set_title("Permutation Feature Importance", fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

def display_risk_thermometer(risk_level):
    st.write("### Risk Level Assessment")
    colors = ["#2ecc71", "#f1c40f", "#f39c12", "#e67e22", "#c0392b"]
    labels = ["1/5", "2/5", "3/5", "4/5", "5/5"]
    descriptions = ["Very Low", "Low", "Moderate", "High", "Critical"]
    cols = st.columns(5)
    
    for i in range(5):
        # On vérifie si c'est le niveau actuel
        is_active = (risk_level == i + 1)
        with cols[i]:
            if is_active:
                st.markdown(
                    f"""
                    <div style="background-color: {colors[i]}; padding: 10px; border-radius: 5px; text-align: center; color: white;">
                        <strong>{labels[i]}</strong><br>{descriptions[i]}
                    </div>
                    """, unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; text-align: center; color: #bdc3c7;">
                        {labels[i]}
                    </div>
                    """, unsafe_allow_html=True
                )

def afficher_grille_risque():
    data = {
        "Level": ["1/5", "2/5", "3/5", "4/5", "5/5"],
        "Probability Range": ["0 - 10%", "10 - 40%", "40 - 60%", "60 - 90%", "90 - 100%"],
        "Status": ["Very Low", "Low", "Moderate", "High", "Critical"]
    }
    df_grid = pd.DataFrame(data)
    st.table(df_grid)

def tracer_graphique_contributions(noms_variables, contributions):
    df_plot = pd.DataFrame({
        'Feature': noms_variables,
        'Contribution': contributions
    })
    df_plot = df_plot.sort_values(by='Contribution', ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    couleurs = ['#80ff80' if val < 0 else '#ff8080' for val in df_plot['Contribution']]
    bordures = ['green' if val < 0 else 'red' for val in df_plot['Contribution']]

    bars = ax.barh(df_plot['Feature'], df_plot['Contribution'], 
                   color=couleurs, edgecolor=bordures, height=0.6)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.2)
        
    for i, bar in enumerate(bars):
        valeur = df_plot['Contribution'][i]
        x_offset = 0.05 if valeur >= 0 else -0.05
        ha_alignment = 'left' if valeur >= 0 else 'right'
        ax.text(valeur + x_offset, bar.get_y() + bar.get_height()/2, 
                f'{valeur:.2f}', 
                va='center', ha=ha_alignment, fontsize=9, fontweight='bold')
        
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.set_xlabel('Contribution to Injury Risk (Log-Odds)', fontsize=12)
    ax.set_title("Individual Risk Profile\n(What drives this player's risk?)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def interactive_prediction(model, feature_names):
    tree = model.tree_
    node = st.session_state.node_index

    if tree.children_left[node] == -1:
        st.success("### Complete Prediction")
        prob = tree.value[node][0][0]
        st.metric("Final Predicted Risk", f"{prob:.4f}")
        st.write(f"**Classification:** {risk_grid(prob)}")
        return

    feature_id = tree.feature[node]
    threshold = tree.threshold[node]
    feat_name = feature_names[feature_id]
    st.subheader(f"Step: Evaluating {feat_name}")
    st.write(f"The model is checking if **{feat_name}** is less than or equal to **{threshold:.2f}**.")
    col1, col2 = st.columns([2, 1])
    with col1:
        val = st.number_input(f"Enter value for {feat_name}:", key=f"input_{node}")
    
    with col2:
        st.write("")
        st.write("")
        if st.button("Validate Step ➡", type="primary"):
            if val <= threshold:
                st.session_state.node_index = tree.children_left[node]
            else:
                st.session_state.node_index = tree.children_right[node]
            st.rerun()

def risk_grid(val): 
    if val <0.1 :
        return "1/5, Really low risk of injury"
    elif val<0.4 :
        return "2/5, Low risk of injury"
    elif val<0.6 :
        return "3/5, Moderate risk of injury"
    elif val<0.9 :
        return "4/5, High risk of injury"
    else :
        return "5/5, Extreme risk of injury"
    
def risk(val): 
    if val <0.1 :
        return 1
    elif val<0.4 :
        return 2
    elif val<0.6 :
        return 3
    elif val<0.9 :
        return 4
    else :
        return 5


def LocGlob():
    if "displocglob" not in st.session_state:
        st.session_state.displocglob ="Logistic Global Interpretation"
    if st.session_state.displocglob =="Logistic Global Interpretation":
        pipeline, colonnes = charger_modele_logistic()
        modele = pipeline.named_steps['lr']
        scaler = pipeline.named_steps['scaler']
        
        st.subheader("Team Summary")
        st.info("This tool aims to give a complete summary and tools to take medical actionnable decision for the entire studied team.")
        st.write("Fill the following CSV file with the value associated to your team. Please do not change the name of the columns before submitting it.")
        
        name_columns=['Age','Height_cm','Weight_kg','Position','Training_Hours_Per_Week', 'Matches_Played_Past_Season', 'Previous_Injury_Count', 'Knee_Strength_Score', 'Hamstring_Flexibility', 'Reaction_Time_ms', 'Balance_Test_Score', 'Sprint_Speed_10m_s', 'Agility_Score', 'Sleep_Hours_Per_Night', 'Stress_Level_Score', 'Nutrition_Quality_Score', 'Warmup_Routine_Adherence', 'BMI']
        df=pd.DataFrame(columns=name_columns)
        csv = df.to_csv(index=False, sep=';').encode('utf-8')
        
        st.download_button(
            label="Download the CSV file to complete",
            data=csv,
            file_name='team_risk.csv',
            mime='text/csv',
        )
        
        st.markdown("Import the data")
        fichier_csv = st.file_uploader("Please drop the file completed", type="csv")
        
        if fichier_csv is not None:
            df_importe = pd.read_csv(fichier_csv, sep=';')
            st.success(f"File uploaded! ({len(df_importe)} players detected)")
            st.divider()
            
            st.subheader("Observed data")
            st.dataframe(df_importe, hide_index=True)
            
            st.subheader("Features statistics")
            with st.expander("Click here to view detailed statistics", expanded=True):
                chosen = st.selectbox("Choose a feature to analyze :", df_importe.columns)
                
                if is_numeric_dtype(df_importe[chosen]) and chosen != 'Warmup_Routine_Adherence':
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean", f"{df_importe[chosen].mean():.2f}")
                        st.metric("Median", f"{df_importe[chosen].median():.2f}")
                    with col2:
                        st.metric("Min", f"{df_importe[chosen].min()}")
                        st.metric("Max", f"{df_importe[chosen].max()}")
                    with col3:
                        st.metric("Std Dev", f"{df_importe[chosen].std():.2f}")

                elif chosen == 'Position':
                    proportions = df_importe[chosen].value_counts(normalize=True)
                    num_cols = min(len(proportions), 4)
                    cols = st.columns(num_cols)
                    for col, (modalite, prop) in zip(cols, proportions.items()):
                        col.metric(f"Pos: {modalite}", f"{prop * 100:.1f} %")

                elif chosen == 'Warmup_Routine_Adherence':
                    n1 = (df_importe[chosen] == 1).sum()
                    n0 = (df_importe[chosen] == 0).sum()
                    col1, col2 = st.columns(2)
                    col1.metric("Without Routine", n0)
                    col2.metric("With Routine", n1)
            

            df_importe['Position'] = df_importe['Position'].str.strip().str.title()
            df_importe = pd.get_dummies(df_importe, dtype=float)
            df_importe = df_importe.reindex(columns=colonnes, fill_value=0.0)

            df_importe_scaled = scaler.transform(df_importe[colonnes])
            df_importe_scaled = pd.DataFrame(df_importe_scaled, columns=colonnes)

            new_col=[]
            for i, col in enumerate(colonnes):
                coefficient_specifique = modele.coef_[0][i]
                df_importe[col + '_contrib'] = df_importe_scaled[col] * coefficient_specifique
                new_col.append(col+'_contrib')
            df_mean = df_importe[new_col].mean().to_frame().T
            
            probabilites_equipe = pipeline.predict_proba(df_importe[colonnes])[:, 1]
            niv = [risk(p) for p in probabilites_equipe]
            
            st.divider()
            st.subheader("Summarized information")
            st.write("Players of the team has been classified in the following classes.")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1: col1.metric(label="Very low risk (1/5)", value=niv.count(1))
            with col2: col2.metric(label="Low risk (2/5)", value=niv.count(2))
            with col3: col3.metric(label="Moderate risk (3/5)", value=niv.count(3))
            with col4: col4.metric(label="High risk (4/5)", value=niv.count(4))
            with col5: col5.metric(label="Very high risk (5/5)", value=niv.count(5))

            valeurs_moyennes = df_mean.iloc[0]
            valeurs_triees = valeurs_moyennes.sort_values(ascending=True)
            top_2_faibles = valeurs_triees.head(2)
            top_2_elevees = valeurs_triees.tail(2)
            
            st.write("The top 2 protecting and risk-increasing factors has been summarized here by averaging the local contribution of each pllayers for each features.")
            col_faible, col_eleve = st.columns(2)
            
            with col_faible:
                st.success("Top 2 Protecting Factors")
                for variable, valeur in top_2_faibles.items():
                    nom_propre = variable.replace("_contrib", "")
                    st.metric(label=nom_propre, value=f"{valeur:.3f}")
                    
            with col_eleve:
                st.error("Top 2 Risk Increasing Factors")
                for variable, valeur in top_2_elevees.sort_values(ascending=False).items():
                    nom_propre = variable.replace("_contrib", "")
                    st.metric(label=nom_propre, value=f"{valeur:.3f}")

        if st.button("Back", key="back_locglob"):
            st.session_state.disp = "Menu Logistic Model"
            st.session_state.displocglob ="Logistic Global Interpretation"
            st.rerun()


def shapval(mod, joueur): 
    explainer = shap.TreeExplainer(mod)
    shap_values = explainer(joueur)
    explication_joueur = shap_values[0, :, 1]
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explication_joueur, max_display=10, show=False)
    plt.tight_layout()
    
    return fig

def Treeloc():
    st.header("Decision Tree Local Interpretation")
    st.info("This tool guides you step-by-step through the model's logic. Answer the questions to reach the final risk prediction.")
    
    if "disptreeloc" not in st.session_state:
        st.session_state.disptreeloc ="Tree Local Interpretation"
    
    if st.session_state.disptreeloc == "Tree Local Interpretation":
        if "node_index" not in st.session_state: 
            st.session_state.node_index = 0
            
        if st.button("Reset Interaction"):
            st.session_state.node_index = 0
            st.rerun()
        
        st.divider()

        with st.container(border=True):
            tree_model = charger_modele_tree()
            colonnes = joblib.load('colonnes_entrainement.pkl')
            interactive_prediction(tree_model, colonnes)

        if st.button("⬅️ Back to Menu"):
            st.session_state.disp ="Menu Logistic Model"
            st.session_state.disptreeloc = "Tree Local Interpretation"
            st.rerun()
            

def LogLoc():
    if "displogloc" not in st.session_state:
        st.session_state.displogloc ="Logistic Local Interpretation"
    if st.session_state.displogloc =="Logistic Local Interpretation":
        st.header("Complete Local Interpretation")
        st.info("This tool aims to provide a local interpretation by quantifying the impact of each features associated to an observation.")
        
        pipeline, colonnes = charger_modele_logistic()
        modele = pipeline.named_steps['lr']
        scaler = pipeline.named_steps['scaler']
        
        with st.expander("Player Information", expanded=True):
            heigth = st.slider("Height in cm:", 150, 210, 177)
            weight = st.slider("Weight in kg:", 40, 110, 73)
            training = st.slider("Total number hours of training per week:", 0.0, 20.0, 10.0)
            position = st.radio("Position played", ["Goalkeeper", "Defender", "Midfielder", "Forward"])
            matches = st.slider("Number of games played past season", 0, 40, 22)
            warmup = st.radio("Warmup routine", ["Yes", "No"])
            injury = st.slider("Previous Injury count:", 0, 10, 0)
            
        with st.expander("Physical Test Score", expanded=True):
            reaction = st.slider("Reaction in ms:", 175.0, 320.0, 250.00)
            st.write(f"Reaction time current value is : {reaction}")
            balance = st.slider("Balance test score:", 0, 100, 50)
            st.write(f"Balance test score current value is : {balance}")
            sprint = st.slider("10m sprint:", 4.50, 7.00, 5.9)
            st.write(f"The time for a 10m sprint current value is : {sprint}")
            hamstring = st.slider("Hamstring Flexibility score:", 0, 100, 50)
            st.write(f"Hamstring flexibility score current value is : {hamstring}")
            knee = st.slider("Knee strength score:", 0, 100, 50)
            st.write(f"Knee strength score current value is : {knee}")
            agility = st.slider("Agility score:", 0, 100, 78)
            st.write(f"Agility score current value is : {agility}")
            
        with st.expander("Life Information", expanded=True):
            stress = st.slider("Stress Level:", 0, 100, 50)
            st.write(f"Stress level current value is : {stress}")
            sleep = st.slider("Sleep hours per night:", 0.00, 12.00, 8.00)
            st.write(f"The number of hours per night current value is : {sleep}")
            nutrition = st.slider("Nutrition Level:", 0, 100, 50)
            st.write(f"Nutrition quality current value is : {nutrition}")

        df_joueur = pd.DataFrame(columns=colonnes)
        df_joueur.loc[0] = 0.0 
        
        df_joueur.at[0, 'Height_cm'] = heigth
        df_joueur.at[0, 'Weight_kg'] = weight
        df_joueur.at[0, 'Training_Hours_Per_Week'] = training
        df_joueur.at[0, 'Matches_Played_Past_Season'] = matches
        df_joueur.at[0, 'Previous_Injury_Count'] = injury
        df_joueur.at[0, 'Knee_Strength_Score'] = knee
        df_joueur.at[0, 'Hamstring_Flexibility'] = hamstring
        df_joueur.at[0, 'Reaction_Time_ms'] = reaction
        df_joueur.at[0, 'Balance_Test_Score'] = balance
        df_joueur.at[0, 'Sprint_Speed_10m_s'] = sprint
        df_joueur.at[0, 'Agility_Score'] = agility
        df_joueur.at[0, 'Sleep_Hours_Per_Night'] = sleep
        df_joueur.at[0, 'Stress_Level_Score'] = stress
        df_joueur.at[0, 'Nutrition_Quality_Score'] = nutrition
        df_joueur.at[0, 'Warmup_Routine_Adherence'] = 1.0 if warmup == "Yes" else 0.0
        df_joueur.at[0, 'BMI'] = weight / ((heigth/100)**2)
        
        col_pos = f"Position_{position}"
        if col_pos in df_joueur.columns:
            df_joueur.at[0, col_pos] = 1.0
            
        probabilite_blessure = pipeline.predict_proba(df_joueur)[0][1]
        niveau_risque = risk_grid(probabilite_blessure)
        
        donnees_standardisees = scaler.transform(df_joueur)
        contribution = donnees_standardisees[0] * modele.coef_[0]
        
        st.header("Result of the model:")
        st.warning(f"Risk level : {niveau_risque}")
        st.write(f"Injury Probability : {probabilite_blessure:.2%}")

        figure = tracer_graphique_contributions(colonnes, contribution)
        st.pyplot(figure)

        if st.button("Back", key="back_logloc"):
            st.session_state.disp ="Menu Logistic Model"
            st.session_state.displogloc = "Logistic Local Interpretation" 
            st.rerun()
        
def rfiloc():
    if "rfiloc" not in st.session_state:
        st.session_state.rfiloc ="Base"
    if st.session_state.rfiloc =="Base":
        st.info("This tool aims to provide a local interpretation by quantifying the impact of each features associated to an observation.")
        modele = joblib.load('modele_rf.pkl')
        colonnes = joblib.load('colonnes_rf.pkl')
        with st.expander("Player Information", expanded=True):
            heigth = st.slider("Height in cm:", 150, 210, 177)
            weight = st.slider("Weight in kg:", 40, 110, 73)
            training = st.slider("Total number hours of training per week:", 0.0, 20.0, 10.0)
            position = st.radio("Position played", ["Goalkeeper", "Defender", "Midfielder", "Forward"])
            matches = st.slider("Number of games played past season", 0, 40, 22)
            warmup = st.radio("Warmup routine", ["Yes", "No"])
            injury = st.slider("Previous Injury count:", 0, 10, 0)
            
        with st.expander("Physical Test Score", expanded=True):
            reaction = st.slider("Reaction in ms:", 175.0, 320.0, 250.00)
            st.write(f"Reaction time current value is : {reaction}")
            balance = st.slider("Balance test score:", 0, 100, 50)
            st.write(f"Balance test score current value is : {balance}")
            sprint = st.slider("10m sprint:", 4.50, 7.00, 5.9)
            st.write(f"The time for a 10m sprint current value is : {sprint}")
            hamstring = st.slider("Hamstring Flexibility score:", 0, 100, 50)
            st.write(f"Hamstring flexibility score current value is : {hamstring}")
            knee = st.slider("Knee strength score:", 0, 100, 50)
            st.write(f"Knee strength score current value is : {knee}")
            agility = st.slider("Agility score:", 0, 100, 78)
            st.write(f"Agility score current value is : {agility}")
            
        with st.expander("Life Information", expanded=True):
            stress = st.slider("Stress Level:", 0, 100, 50)
            st.write(f"Stress level current value is : {stress}")
            sleep = st.slider("Sleep hours per night:", 0.00, 12.00, 8.00)
            st.write(f"The number of hours per night current value is : {sleep}")
            nutrition = st.slider("Nutrition Level:", 0, 100, 50)

        df_joueur = pd.DataFrame(columns=colonnes)
        df_joueur.loc[0] = 0.0 
        
        df_joueur.at[0, 'Height_cm'] = heigth
        df_joueur.at[0, 'Weight_kg'] = weight
        df_joueur.at[0, 'Training_Hours_Per_Week'] = training
        df_joueur.at[0, 'Matches_Played_Past_Season'] = matches
        df_joueur.at[0, 'Previous_Injury_Count'] = injury
        df_joueur.at[0, 'Knee_Strength_Score'] = knee
        df_joueur.at[0, 'Hamstring_Flexibility'] = hamstring
        df_joueur.at[0, 'Reaction_Time_ms'] = reaction
        df_joueur.at[0, 'Balance_Test_Score'] = balance
        df_joueur.at[0, 'Sprint_Speed_10m_s'] = sprint
        df_joueur.at[0, 'Agility_Score'] = agility
        df_joueur.at[0, 'Sleep_Hours_Per_Night'] = sleep
        df_joueur.at[0, 'Stress_Level_Score'] = stress
        df_joueur.at[0, 'Nutrition_Quality_Score'] = nutrition
        df_joueur.at[0, 'Warmup_Routine_Adherence'] = 1.0 if warmup == "Yes" else 0.0
        df_joueur.at[0, 'BMI'] = weight / ((heigth/100)**2)
        
        col_pos = f"Position_{position}"
        if col_pos in df_joueur.columns:
            df_joueur.at[0, col_pos] = 1.0
        
        probabilite_blessure = modele.predict_proba(df_joueur)[0][1]
        niveau_risque = risk_grid(probabilite_blessure)
        
        
        st.header("Result of the model:")
        st.warning(f"Risk level : {niveau_risque}")
        st.write(f"Injury Probability : {probabilite_blessure:.2%}")
        fig_shap = shapval(modele, df_joueur)
        st.pyplot(fig_shap)
        if st.button("Back", key="back_rfiloc"):
            st.session_state.disprf ="Menu RF Model"
            st.rerun()

def rfteam():
    if "rfteam" not in st.session_state:
        st.session_state.rfteam ="Base"
    if st.session_state.rfteam =="Base":
        modele = joblib.load('models/modele_rf.pkl')
        colonnes = joblib.load('models/colonnes_rf.pkl')
        
        st.subheader("Team Summary")
        st.info("This tool aims to give a complete summary and tools to take medical actionnable decision for the entire studied team.")
        st.write("Fill the following CSV file with the value associated to your team. Please do not change the name of the columns before submitting it.")
        
        name_columns=['Age','Height_cm','Weight_kg','Position','Training_Hours_Per_Week', 'Matches_Played_Past_Season', 'Previous_Injury_Count', 'Knee_Strength_Score', 'Hamstring_Flexibility', 'Reaction_Time_ms', 'Balance_Test_Score', 'Sprint_Speed_10m_s', 'Agility_Score', 'Sleep_Hours_Per_Night', 'Stress_Level_Score', 'Nutrition_Quality_Score', 'Warmup_Routine_Adherence', 'BMI']
        df=pd.DataFrame(columns=name_columns)
        csv = df.to_csv(index=False, sep=';').encode('utf-8')
        
        st.download_button(
            label="Download the CSV file to complete",
            data=csv,
            file_name='team_risk.csv',
            mime='text/csv',
        )
        
        st.markdown("Import the data")
        fichier_csv = st.file_uploader("Please drop the file completed", type="csv")
        
        if fichier_csv is not None:
            df_importe = pd.read_csv(fichier_csv, sep=';')
            st.success(f"File uploaded! ({len(df_importe)} players detected)")
            st.divider()
            
            st.subheader("Observed data")
            st.dataframe(df_importe, hide_index=True)
            
            st.subheader("Features statistics")
            with st.expander("Click here to view detailed statistics", expanded=True):
                chosen = st.selectbox("Choose a feature to analyze :", df_importe.columns)
                
                if is_numeric_dtype(df_importe[chosen]) and chosen != 'Warmup_Routine_Adherence':
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean", f"{df_importe[chosen].mean():.2f}")
                        st.metric("Median", f"{df_importe[chosen].median():.2f}")
                    with col2:
                        st.metric("Min", f"{df_importe[chosen].min()}")
                        st.metric("Max", f"{df_importe[chosen].max()}")
                    with col3:
                        st.metric("Std Dev", f"{df_importe[chosen].std():.2f}")

                elif chosen == 'Position':
                    proportions = df_importe[chosen].value_counts(normalize=True)
                    num_cols = min(len(proportions), 4)
                    cols = st.columns(num_cols)
                    for col, (modalite, prop) in zip(cols, proportions.items()):
                        col.metric(f"Pos: {modalite}", f"{prop * 100:.1f} %")

                elif chosen == 'Warmup_Routine_Adherence':
                    n1 = (df_importe[chosen] == 1).sum()
                    n0 = (df_importe[chosen] == 0).sum()
                    col1, col2 = st.columns(2)
                    col1.metric("Without Routine", n0)
                    col2.metric("With Routine", n1)
            
            df_importe['Position'] = df_importe['Position'].str.strip().str.title()
            df_importe = pd.get_dummies(df_importe, dtype=float)
            df_importe = df_importe.reindex(columns=colonnes, fill_value=0.0)

            probabilites_equipe = modele.predict_proba(df_importe[colonnes])[:, 1]
            niv = [risk(p) for p in probabilites_equipe]
            
            st.divider()
            st.subheader("Summarized information")
            st.write("Players of the team have been classified in the following risk classes:")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1: col1.metric(label="Very low risk (1/5)", value=niv.count(1))
            with col2: col2.metric(label="Low risk (2/5)", value=niv.count(2))
            with col3: col3.metric(label="Moderate risk (3/5)", value=niv.count(3))
            with col4: col4.metric(label="High risk (4/5)", value=niv.count(4))
            with col5: col5.metric(label="Very high risk (5/5)", value=niv.count(5))

            st.write("The top 2 protecting and risk-increasing factors have been summarized here by averaging the local SHAP contributions of all players.")
            
            explainer = shap.TreeExplainer(modele)
            shap_values = explainer(df_importe[colonnes])
            shap_array = shap_values.values[:, :, 1] 
            shap_moyens = np.mean(shap_array, axis=0)
            serie_shap = pd.Series(shap_moyens, index=colonnes)
            serie_shap_trie = serie_shap.sort_values(ascending=True)
            top_2_faibles = serie_shap_trie.head(2)
            top_2_elevees = serie_shap_trie.tail(2)
            col_faible, col_eleve = st.columns(2)
            
            with col_faible:
                st.success("Top 2 Protecting Factors (Negative SHAP)")
                for variable, valeur in top_2_faibles.items():
                    nom_propre = variable.replace("_", " ") 
                    st.metric(label=nom_propre, value=f"{valeur:.2%}")
                    
            with col_eleve:
                st.error("Top 2 Risk Increasing Factors (Positive SHAP)")
                for variable, valeur in top_2_elevees.sort_values(ascending=False).items():
                    nom_propre = variable.replace("_", " ")
                    st.metric(label=nom_propre, value=f"{valeur:.2%}")
        if st.button("Back", key="back_rfteam"):
            st.session_state.disprf = "Menu RF Model"
            st.session_state.rfteam = "Base"
            st.rerun()


def Logistic():
    if "disp" not in st.session_state:
        st.session_state.disp ="Menu Logistic Model"

    if st.session_state.disp =="Menu Logistic Model":
        pipeline, colonnes = charger_modele_logistic()
        modele = pipeline.named_steps['lr']
        st.subheader("Logistic Regression Model")
        st.write("The tools developed in this section are all based on the application of the logistic regression model to the dataset presented on the home page. This model achieves a high overall accuracy of 95.2% and proves to be very effective for both global and local interpretation, as presented below.")
        st.divider()
        st.subheader("Risk Classification Grid")
        st.write("The main problem for medical staff with these models lies in the way players are classified into a binary category. This classification is too simplistic and groups together players who do not have much in common. In fact, medical staff will not treat a player with a 90% risk of injury the same way as one with a 55% risk, since the latter has much more in common with a player who has a 45% risk.")
        st.write("That is why a different risk classification grid has been implemented in the following section, based on the following logic:")
        afficher_grille_risque()
        st.divider()
        st.subheader("Global Interpretation")
        st.write("The overall interpretation can be based on the odds ratio. Specifically, the odds ratio quantifies the strength of the association between a characteristic and the target variable by indicating the proportional change in the odds of the outcome for each one-unit increase in that characteristic.")
        st.write("The odds ratio associated to each features are summarized in the following table. Some variables were not included because the estimates were not significant.")
        odds_ratios = np.exp(modele.coef_[0])
        df_or = pd.DataFrame({
            'Feature': colonnes,
            'Odds Ratio': odds_ratios
        })
        df_or = df_or.sort_values(by='Odds Ratio', ascending=False)
        df_or = df_or[df_or["Odds Ratio"]!=1]
        # On ajoute une colonne d'interprétation pour aider l'utilisateur
        df_or['Interpretation'] = df_or['Odds Ratio'].apply(
            lambda x: "📈 Increase Risk" if x > 1 else "🟢 Decrease Risk"
        )
        st.dataframe(
            df_or.style.background_gradient(cmap='RdYlGn_r', subset=['Odds Ratio']), # Note le _r à la fin
            use_container_width=True,
            hide_index=True
        )
        st.divider()
        st.subheader("Local Interpretation")
        st.write("Regarding local interpretation, various methods are proposed below.")
        st.write("The first is based on the entire dataset and calculates the impact of each variable on the probability of injury associated with an observation. After the estimation of the parameters of the Logistic Regression Model, we can easily calulate the impact of each variable by multiplying this coefficient to the centered and scaled value.")
        if st.button("Complete Local Interpretation"):
                st.session_state.disp = "3_1"
                st.rerun()
        st.write("The second tool was created by fitting a decision tree to the output probabilities calculated by the logistic regression model. In this case, fitting a decision tree resulted in a loss of overall accuracy, but it allowed us to reduce the number of features to three while maintaining an accurate result, instead of having to pass 18 tests. ")
        if st.button("Decision tree based interpretation"):
                st.session_state.disp = "3_2"
                st.rerun()
        st.divider()
        st.subheader("Overall Team Interpretation")
        st.write("Finally, a team-based interpretation tool has also been implemented. The idea is to give medical staff the ability to identify the characteristics that have the greatest impact on the team by calculating the average of each characteristic’s overall local contribution. This section also provides a comprehensive summary of each characteristic. ")
        if st.button("Team interpretation"):
                st.session_state.disp = "2"
                st.rerun()

        if st.button("Back", key="back_menu_log"):
            st.session_state.dispositif="Welcome"
            st.session_state.disp = "Menu Logistic Model" 
            st.rerun()
    elif st.session_state.disp == "2":
        LocGlob()
    elif st.session_state.disp == "3_1":
        LogLoc()
    elif st.session_state.disp == "3_2":
        Treeloc()


def RandomF():
    if "disprf" not in st.session_state:
        st.session_state.disprf ="Menu RF Model"
    if st.session_state.disprf =="Menu RF Model":
        st.subheader("Random Forest Model")
        st.write("The tools developed in this section are based on the application of a Random Forest classifier to the dataset presented on the home page. As an advanced ensemble learning method, this model excels at capturing complex, non-linear relationships between physical capabilities and lifestyle habits. It achieves an overall test accuracy of 94.3%, it slightly underperforms the Logistic Regression baseline. However, in the context of medical decision-making, the ultimate goal is not merely maximizing a performance metric, but rather understanding complex, non-linear relationships between physical capabilities and lifestyle habits.")
        st.write("However, unlike Logistic Regression, a Random Forest's architecture relies on hundreds of decision trees, creating a 'Black Box' effect that prevents direct reading of coefficients. To overcome this limitation and provide actionable insights for medical staff, we integrated advanced eXplainable AI (XAI) techniques, achieving a high level of transparency as presented below.")
        st.divider()
        st.subheader("Risk Classification Grid")
        st.write("The main problem for medical staff with these models lies in the way players are classified into a binary category. This classification is too simplistic and groups together players who do not have much in common. In fact, medical staff will not treat a player with a 90% risk of injury the same way as one with a 55% risk, since the latter has much more in common with a player who has a 45% risk.")
        st.write("That is why a different risk classification grid has been implemented in the following section, based on the following logic:")
        afficher_grille_risque()
        st.divider()
        st.subheader("Global Interpretation")
        st.write("The overall interpretation of the model can be based on the two new techniques which are the Permutation Features Importance (PFI), the Partial Dependance (PD) and the Individually Contribution Expectation (ICE). Specifically, the PFI aims to quantify the importance of each feature in the model by removing the information of a feature and calculate estimate the decrease in the accuracy. As this tool doesn't provide an information concerning the impact (increasing, decreasing...), it's completed by the PD and the ICE techniques. Their goals are to display the average (PD) and the individual (ICE) evolution of the predict probability across the range of a studied feature.")
        st.write("In this section, the result are based on a 80/20 split as we need a testing set to calculate these values.")
        st.write("The following waterfall graphic displays the result of the Permutation Feature Importance. Only features that actively contribute to the model's performance are shown.")
        afficher_pfi()
        st.write("THe following section displayed the ICE and PD plot of the variables that seemed relevant on the above PFI plot.")
        with st.expander("Click here to view detailed ICE and PD", expanded=True):
            chosen = st.selectbox("Choose a feature to analyze :", ['Stress_Level_Score', 'Sleep_Hours_Per_Night', 'Reaction_Time_ms', 'Balance_Test_Score', 'Knee_Strength_Score', 'Sprint_Speed_10m_s', 'Hamstring_Flexibility', 'Nutrition_Quality_Score', 'Agility_Score', 'Previous_Injury_Count', 'Height_cm', 'Weight_kg', 'BMI', 'Warmup_Routine_Adherence'])
            chemin_image = f"ICE\ICE_{chosen}.png"
            st.image(chemin_image, caption=f"ICE & PDP plot for {chosen}")
        st.divider()
        st.subheader("Individual Interpretation")
        st.write("Regarding individual interpretation, the following tool is based on the SHAP value and aims at providing a risk prediction of injury and provide an explanation of the result.")
        if st.button("Individual Interpretation"):
            st.session_state.disprf="RF_ILoc"
            st.rerun()
        st.divider()
        st.subheader("Team Interpretation")
        st.write("This tool aims at explaining the to give medical staff the ability to identify the characteristics that have the greatest impact on the team by calculating the average of each characteristic’s overall local contribution. This section also provides a comprehensive summary of each characteristic. ")
        if st.button("Team Interpretation"):
            st.session_state.disprf="RF_Team"
            st.rerun()
        if st.button("Back", key="back_menu_rf"):
            st.session_state.dispositif="Welcome"
            st.session_state.disprf = "Menu RF Model" 
            st.rerun()
    if st.session_state.disprf=="RF_ILoc":
        rfiloc()
    if st.session_state.disprf == "RF_Team":
        rfteam()
    
        
        
def Methodology():
    st.title("Dataset Methodology & Description")
    st.divider()

    st.subheader("A.1.1 Demographical Information")
    st.markdown("""
    Player information is summarized in 8 variables (1 nominal-qualitative, 3 quantitative discrete, 4 continuous quantitative):
    - **Age**: Age of the football player, from 18 to 24.
    - **Position**: The position the player played in his team (*goalkeeper, defender, midfielder, forward*).
    - **Previous Injury Count**: Number of injuries already suffered.
    - **Height, Weight, BMI**: Anthropometric measurements and the Body Mass Index.
    - **Training Hours per Week**: Player’s weekly training load.
    - **Matches played past season**: Number of games participated in last season.
    """)

    st.write("") 
    
    st.subheader("A.1.2 Physical strength and abilities")
    st.markdown("""
    Player strength tests and abilities are captured by 6 continuous variables:
    - **Knee Strength Score**: Score (0-100) based on muscle strength near the knee (Quadriceps, Hamstring, Calves).
    - **Hamstring Flexibility**: Score (0-100) expressing hamstring flexibility.
    - **Balance Test**: Score (0-100) expressing player balance.
    - **Agility score**: Score (0-100) expressing player agility.
    - **Reaction Time**: Expressed in milliseconds (ms).
    - **Sprint 10m**: Time to complete 10 meters sprint (seconds).
    """)

    st.write("")

    st.subheader("A.1.3 Life Habits")
    st.markdown("""
    Life habits are captured by 4 variables:
    - **Sleep hours per night**: Average nocturnal recovery duration (e.g., 7.5).
    - **Stress level score**: Score (0-100). Higher score indicates significant mental fatigue.
    - **Nutrition quality score**: Score (0-100) based on dietary balance. 100 = perfectly tailored diet.
    - **Warmup routine**: Binary variable (1: Routine completed, 0: Routine incomplete).
    """)

    st.divider()
    
    if st.button("Back to Home"):
        st.session_state.dispositif = "Welcome"
        st.rerun()

def main():
    st.set_page_config(page_title="Injury Prediction", page_icon="", layout="wide")

    st.title("Injury Prediction Dashboard")
    st.divider()
    with st.sidebar:
        st.title("Project Info")
        st.write("---")
        st.subheader("Navigation")
        if st.button("Back to Welcome Page"):
            reinit()
            st.session_state.dispositif = "Welcome"
        if st.button("Context Description"):
            reinit()
            st.session_state.dispositif = "3"
        if st.button("Model 1 - Logistic Regression"):
            reinit()
            st.session_state.dispositif = "1"
        if st.button("Model 2 - Random Forest"):
            reinit()
            st.session_state.dispositif = "2"
        st.write("---")
        st.subheader("Dataset Dictionary")
        
        with st.expander("Demographical Information"):
            st.markdown("""
            * **Age**: 18 to 24.
            * **Position**: Role in the team.
            * **Prev. Injury**: Career injury count.
            * **Physicals**: Height, Weight, BMI.
            * **Training**: Weekly load (hours).
            * **Matches**: Games played last season.
            """)
            
        with st.expander("Life Habits"):
            st.markdown("""
            * **Sleep**: Average hours/night.
            * **Stress**: Score (0-100).
            * **Nutrition**: Score (0-100).
            * **Balance**: Score (0-100).
            * **Warmup**: Routine status (1: Yes, 0: No).
            """)
            
        with st.expander(" Physical Tests"):
            st.markdown("""
            * **Knee Strength**: Score (0-100).
            * **Hamstring Flex**: Score (0-100).
            * **Balance/Agility**: Score (0-100).
            * **Reaction Time**: In ms.
            * **Sprint 10m**: Time in seconds.
            """)
        if st.button("Further explanations"):
            st.session_state.dispositif ="4"
        st.divider()
        st.caption("Tohoku University - Statistical Lab")
        st.caption("Jules VOEGTLE - 1st year of Master")
        st.caption("voegtlejules@gmail.com")
    if "dispositif" not in st.session_state:
        st.session_state.dispositif = "Welcome"
    if  st.session_state.dispositif == "Welcome":
        st.write("### Welcome to the website's homepage.")
            
        st.markdown("""
        The purpose of this site is to provide an interactive way to predict injuries among professional 
        soccer players at the start of the season by administering a battery of physical and other tests. 
        It also aims to offer an interactive way to interpret these predictions.
            
        Two different models are presented here:
        * **Model 1:** Based on a **Logistic Regression** model.
        * **Model 2:** Based on an “ensemble” method known as **Random Forest**.
        """)

            # --- Section des Modèles ---
        st.subheader("Select a Model to start")
        col1, col2 = st.columns(2)
            
        with col1:
            if st.button("Model 1 - Logistic Regression"):
                st.session_state.dispositif = "1"
                st.rerun()
                    
        with col2:
            if st.button("Model 2 - Random Forest"):
                st.session_state.dispositif = "2"
                st.rerun()

            # --- Section Disclaimer et Info ---
        st.divider()
            
        st.info("""
            **Disclaimer:** The model is based on a synthetic dataset and therefore does not 
            constitute a real forecasting tool; rather, it serves as a basis for a semester-long 
            research project.
            """)

            # --- Section Context / Data ---
        col3, col4 = st.columns(2)
        with col3:
            if st.button("Context of the Study"):
                st.session_state.dispositif = "3" # Ton état pour le contexte
                st.rerun()
                    
        with col4:
            if st.button("Dataset"):
                st.session_state.dispositif = "4" # Nouvel état à créer
                st.rerun()


    elif st.session_state.dispositif =="1":
        Logistic()

    elif st.session_state.dispositif =="2":
        RandomF()
    elif st.session_state.dispositif =="3":
        st.write("The tools developped in this web application serve as a proof of concept for Machine Learning using in the context of sport injury prediction.  ")
        st.write("Two model have been used, the first one is the Logistic Regression Model and has been used as a bench mark model. Indeed, the objective of this study is to apply and implement different tools to reach this level of interpretability for other models such as the Random Forest. Eventhough in this case, Logistic Regression model manage to have a great performance by catching the pattern of this dataset, ensemble methods often outperform the Logistic Regression Model in the context of sport injury prediction. ")
        st.write("This web application was created during a research project conducted in the Statistical Lab of the Graduate School of Information Sciences of the Tohoku University. The objective was to propose an interactive representation of the principles presented in this research.")
        if st.button("Back Home"):
            st.session_state.dispositif = "Welcome"
    elif st.session_state.dispositif =="4":
        Methodology()

main()
