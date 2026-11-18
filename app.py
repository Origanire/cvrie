"""CVRIE — local research demonstrator, never a diagnostic device."""
import json

import pandas as pd
import streamlit as st

from src.common.paths import DATA, MODEL, REPORT
from src.supervised.inference import load_model, predict
from src.supervised.preprocess import read_image

st.set_page_config(page_title="CVRIE · Analyse thoracique", page_icon="🫁", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container {max-width:1200px; padding-top:2.4rem;}
h1 {letter-spacing:-.045em; font-weight:750 !important;}
.hero {padding:24px 28px;border-radius:20px;background:#142D3D;color:white;margin-bottom:24px;}
.hero p {color:#c6d8e2;margin-bottom:0;max-width:770px;}
.eyebrow {font-size:12px;letter-spacing:.16em;color:#85ded8;margin-bottom:12px;}
.step {font-size:12px;letter-spacing:.12em;color:#087F8C;font-weight:700;margin:12px 0;}
[data-testid="stMetric"] {background:white;padding:16px;border-radius:12px;border:1px solid #dce5eb;}
</style>
<div class="hero"><div class="eyebrow">CVRIE / LABORATOIRE D’IMAGERIE</div>
<h1>Explorer une radiographie thoracique.</h1>
<p>Importez une image et obtenez une classification expérimentale en quelques instants.
Un projet de recherche sur les nodules pulmonaires, fondé sur des radiographies publiques.</p></div>
""", unsafe_allow_html=True)
st.warning("Prototype pédagogique — aucune utilisation diagnostique. Le résultat ne permet pas de savoir si vous avez un cancer. Un médecin doit interpréter votre examen.")


@st.cache_resource
def cached_model(modified_ns):
    return load_model()


try:
    bundle = cached_model(MODEL.stat().st_mtime_ns)
except FileNotFoundError:
    st.info("Le modèle n'est pas encore entraîné. Préparez les données puis lancez l'entraînement :")
    st.code("python -m src.supervised.dataset\npython -m src.supervised.improve", language="bash")
    st.stop()
except Exception:
    st.error("Impossible de charger le modèle local. Réentraînez-le avec la version actuelle du projet.")
    st.stop()

report = bundle["report"]
with st.sidebar:
    st.markdown("## CVRIE")
    st.caption("IMAGERIE · APPRENTISSAGE SUPERVISÉ")
    st.divider()
    st.markdown("**Analyse locale**")
    st.caption("Les images importées sont traitées en mémoire par ce serveur local et ne sont pas enregistrées par l'application.")
    st.markdown("**Périmètre**")
    st.caption("Radiographie thoracique de face · PNG ou JPEG · niveaux de gris · 20 Mo maximum.")
    st.markdown("**Modèle actif**")
    st.code(report["selected_model"], language=None)
    st.caption("Pas de scanner, d'IRM ou de photographie d'écran. Le modèle ne sait pas vérifier l'anatomie de l'image.")
    st.divider()
    st.markdown("[Base JSRT](http://db.jsrt.or.jp/eng.php) · [Images JSRT](https://www.kaggle.com/datasets/raddar/nodules-in-chest-xrays-jsrt)")

analysis_tab, evaluation_tab, data_tab = st.tabs(["Analyser une image", "Performances du modèle", "Données et méthode"])
with analysis_tab:
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown('<div class="step">01 / CHOISIR UNE RADIOGRAPHIE</div>', unsafe_allow_html=True)
        source = st.radio("Source de l'image", ["Importer mon image", "Essayer un exemple du jeu de test"], horizontal=True)
        content = None
        example_truth = None
        if source == "Importer mon image":
            uploaded = st.file_uploader("Importer une radiographie", type=["png", "jpg", "jpeg"],
                                        help="Export 8 bits sans informations nominatives. DICOM non pris en charge.")
            if uploaded:
                content = uploaded.getvalue()
        else:
            manifest_path = DATA / report.get("manifest_filename", "manifest_2048.csv" if bundle.get("schema_version") == 2 else "manifest.csv")
            if manifest_path.exists():
                frame = pd.read_csv(manifest_path)
                examples = frame[frame.split == "Test"]
                choice = st.selectbox("Exemple réservé à l'évaluation", examples.patient_id.tolist())
                row = examples[examples.patient_id == choice].iloc[0]
                path = (DATA / row.path).resolve()
                if path.is_relative_to(DATA.resolve()) and path.is_file():
                    content = path.read_bytes()
                    example_truth = {"malignant": "nodule malin", "benign": "nodule bénin", "non-nodule": "sans nodule"}[row.state]
                else:
                    st.error("Image d'exemple absente. Relancez la préparation des données.")
            else:
                st.info("Téléchargez la base pour accéder aux exemples.")
        valid = False
        if content:
            try:
                image = read_image(content)
                st.image(image, caption=f"Image fournie · {image.width} × {image.height} px", width="stretch")
                valid = True
            except ValueError as exc:
                st.error(str(exc))
        else:
            st.info("Glissez votre image dans la zone d'import ou choisissez un exemple pour découvrir le fonctionnement.")
    with right:
        st.markdown('<div class="step">02 / LANCER L’ANALYSE</div>', unsafe_allow_html=True)
        st.subheader("Une prédiction, avec ses limites")
        st.write("Le modèle compare les caractéristiques de l'image aux cas de sa base d'entraînement. Il n'établit pas de diagnostic.")
        confirmed = st.checkbox("Radio thoracique de face · usage expérimental",
                                key="confirm_" + source)
        analyze = st.button("Analyser la radiographie", type="primary", width="stretch", disabled=not (valid and confirmed))
        if analyze:
            with st.spinner("Analyse locale de l'image…"):
                try:
                    result = predict(content, bundle)
                except (ValueError, OSError):
                    st.error("L'analyse a échoué. Vérifiez le fichier et le modèle local.")
                else:
                    st.markdown("### Résultat expérimental")
                    st.info(result["label"])
                    st.metric("Score du modèle pour la classe « annotation maligne »", f"{result['score']:.3f} / 1")
                    st.progress(result["score"])
                    st.caption(f"Ce score n'est pas votre probabilité d'avoir un cancer. Seuil de classification : {result['threshold']:.3f}, choisi sur validation pour le modèle amélioré, non validé cliniquement.")
                    st.write(result["interpretation"])
                    if result["known_dataset_split"]:
                        st.caption(f"Image reconnue dans la base : jeu {result['known_dataset_split']}. Ce n'est pas une validation sur une nouvelle population.")
                    if example_truth:
                        st.write(f"**Annotation du jeu de test :** {example_truth}.")
                    st.caption(f"Temps de calcul : {result['elapsed_ms']} ms · modèle {result['model']}")
                    st.download_button("Télécharger le résultat JSON", json.dumps(result, indent=2, ensure_ascii=False),
                                       file_name="cvrie-resultat-experimental.json", mime="application/json")
        st.divider()
        st.markdown("**À retenir**")
        st.write("Un classement « bénin ou sans nodule » n'exclut pas un cancer. Un classement « nodule malin » ne le confirme pas. Ne modifiez pas votre suivi médical sur cette base.")
        st.caption(f"Évaluation interne : {report['test']['n']} images · AUC {report['test']['roc_auc']:.2f}. Consultez l'onglet Performances pour les erreurs observées.")

with evaluation_tab:
    st.subheader("Ce que le modèle a réellement réussi")
    st.write(f"{report['split_counts']['Train']} images d'entraînement, {report['split_counts']['Validation']} de validation et {report['test']['n']} de test. Le test a déjà été consulté : les résultats sont exploratoires, sans validation clinique indépendante.")
    if "baseline" in report:
        st.markdown("**Comparaison avec la version précédente — mêmes patients de test**")
        comparison_rows = {}
        for label, key in [("AUC ROC", "roc_auc"), ("Exactitude", "accuracy"), ("Précision positive", "precision"), ("Sensibilité", "sensitivity"), ("Spécificité", "specificity")]:
            comparison_rows[label] = {"Avant": report["baseline"]["test"][key], "Maintenant": report["test"][key]}
        st.dataframe(pd.DataFrame(comparison_rows).T.round(3), width="stretch")
        st.caption("Le seuil a changé : la comparaison des décisions inclut le nouvel encodeur et le réglage du seuil. L'AUC ne dépend pas du seuil.")
    metrics = report["test"]
    cols = st.columns(4)
    for col, label, key in zip(cols, ["AUC ROC", "Sensibilité", "Spécificité", "Exactitude"], ["roc_auc", "sensitivity", "specificity", "accuracy"]):
        col.metric(label, f"{metrics[key]:.3f}")
    lo, hi = metrics["roc_auc_ci95_bootstrap"]
    st.caption(f"Intervalle bootstrap à 95 % pour l'AUC : [{lo:.2f} ; {hi:.2f}]. AUC = 0,50 correspond à un classement aléatoire. Petit échantillon : estimation instable.")
    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]
    st.error(f"Sur ce test : {fn} cas malins manqués et {fp} fausses alertes. Ce prototype n'est pas validé pour une décision médicale.")
    st.markdown("**Matrice de confusion — lignes : annotation réelle, colonnes : prédiction**")
    st.dataframe(pd.DataFrame([[tn, fp], [fn, tp]],
        index=["Bénin / sans nodule", "Malin"], columns=["Prédit bénin / sans nodule", "Prédit malin"]), width="stretch")
    st.caption(f"Référence naïve « toujours négatif » : {report['test_majority_baseline_accuracy']:.0%} d'exactitude.")
    if "cross_validation" in report:
        st.markdown("**Sélection par validation croisée sur l'entraînement (5 plis ; répétitions indiquées dans le rapport)**")
        st.dataframe(pd.DataFrame(report["cross_validation"]).T[["auc_mean", "auc_std"]].astype(float).round(3), width="stretch")
        st.caption("Configurations fixées à l'avance ; choix par AUC moyenne sur l’entraînement. Le seuil est ensuite réglé sur Validation, sans optimiser sur Test.")
    st.markdown("**Mesures sur la validation (seuil réglé sur ce jeu)**")
    comparison = pd.DataFrame(report["validation"]).T
    st.dataframe(comparison[["roc_auc", "sensitivity", "specificity", "accuracy", "train_seconds"]].astype(float).round(3), width="stretch")
    st.download_button("Télécharger le rapport d'évaluation", json.dumps(report, indent=2, ensure_ascii=False),
                       file_name="cvrie-evaluation.json", mime="application/json")

with data_tab:
    extended_path = DATA / "manifest_extended.csv"
    available = pd.read_csv(extended_path if extended_path.exists() else DATA / "manifest_2048.csv")
    st.subheader(f"{len(available)} radiographies uniques disponibles")
    if "source" in available:
        st.dataframe(available.groupby(["source", "split"]).size().unstack(fill_value=0), width="stretch")
    st.write(f"Le modèle actif a été entraîné sur {report['split_counts']['Train']} patients. Sa cohorte complète contient {report['dataset_size']} images, validation et test compris.")
    st.write("Une image par patient. Les patients JSRT conservent leur séparation initiale ; les nouveaux patients sont répartis avant l’entraînement. Les empreintes des fichiers et les identifiants sont contrôlés.")
    if extended_path.exists():
        st.write("LIDC-IDRI : radiographies publiques de face DX. Cas positifs : cancer pulmonaire primitif confirmé par biopsie ou résection. Cas négatifs : diagnostic bénin établi par stabilité sur au moins deux ans, biopsie ou résection. Les cas inconnus, métastatiques et fondés uniquement sur la progression sont exclus.")
        st.warning("LIDC fournit ici un diagnostic au niveau du patient, sans garantir que la lésion est visible sur cette radio ou que l’examen correspond à la date du diagnostic. L’association avec les annotations de nodules JSRT reste expérimentale.")
        st.caption("LIDC-IDRI : Armato et al., TCIA, licence CC BY 3.0. Conversion DICOM en niveaux de gris avec normalisation fixe par percentiles ; une vue frontale documentée par patient.")
        st.markdown("[Collection publique et attribution LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/)")
    st.write("JSRT : images PNG 2048 × 2048 du miroir public raddar. 246 patients après exclusion du doublon de la base initiale. Les versions de meilleure résolution ne sont pas comptées comme de nouveaux patients.")
    st.write("Les encodeurs thoraciques préentraînés extraient les caractéristiques ; leurs poids restent gelés. Les classifieurs apprennent uniquement sur le jeu d’entraînement. Le seuil est choisi sur Validation. Les résultats Test ne servent pas à choisir le modèle.")
    st.markdown("**Limites du modèle actuellement actif**")
    for limitation in report["limitations"]:
        st.markdown(f"- {limitation.replace('aucun patient supplémentaire ni validation clinique externe', 'les patients ajoutés sont utilisés par les candidats, sans validation clinique externe')}")
    st.markdown("[Images JSRT](https://www.kaggle.com/datasets/raddar/nodules-in-chest-xrays-jsrt)")

candidate_path = REPORT.parent / "candidate_v4.json"
if not candidate_path.exists():
    candidate_path = REPORT.parent / "candidate_v3.json"
if candidate_path.exists():
    with evaluation_tab:
        extended = json.loads(candidate_path.read_text())
        st.divider()
        st.subheader("Dernier entraînement avec les nouveaux patients")
        st.caption(f"Configuration : {extended['selected_model']}")
        if candidate_path.name == "candidate_v4.json":
            st.write("Bilan des deux cycles : 40 configurations et 600 ajustements de validation croisée, avec encodeurs 224 px et 512 px.")
        st.write(f"{extended['dataset_size']} patients au total · 20 configurations · 5 plis répétés 3 fois · 300 ajustements de validation croisée.")
        if extended["promoted"]:
            st.success("Le modèle élargi a satisfait les critères de validation et est actif.")
        else:
            st.info("Le modèle élargi a été entraîné et sauvegardé. Il n’a pas satisfait tous les critères de validation : le modèle précédent reste actif.")
        st.caption("Décision prise avant le calcul des scores Test : amélioration de l’AUC et de l’exactitude équilibrée moyennes par source sur Validation, sans recul de l’AUC JSRT.")
        rows = []
        for source, measures in extended["test_by_source"].items():
            for version, values in [("Précédent", extended["incumbent_test_by_source"][source]), ("Élargi", measures)]:
                rows.append({"Source": source, "Modèle": version, "Patients test": values["n"], "AUC": values["roc_auc"], "Exactitude": values["accuracy"], "Sensibilité": values["sensitivity"], "Spécificité": values["specificity"]})
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
        st.caption("JSRT : test déjà consulté. LIDC : très petit test de nouveaux patients, maintenant consulté. Ces scores ne démontrent pas une fiabilité clinique.")
        st.download_button("Télécharger le bilan de l’entraînement élargi", json.dumps(extended, indent=2, ensure_ascii=False), file_name="cvrie-entrainement-elargi.json", mime="application/json")
