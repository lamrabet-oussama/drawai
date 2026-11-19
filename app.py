"""DrawAI - handwritten digit recognition with scikit-learn and Streamlit."""

import plotly.graph_objects as go
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from src.prediction import is_blank, load_model, predict

CANVAS_SIZE = 280

st.set_page_config(page_title="DrawAI", page_icon="✏️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1100px; padding-top: 2rem;}
    .digit-box {text-align: center; padding: 1rem; border-radius: 12px;
                background: rgba(128,128,128,0.12);}
    .digit-box .digit {font-size: 7rem; font-weight: 700; line-height: 1;}
    .digit-box .caption {opacity: 0.75; font-size: 0.95rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Chargement du modèle…")
def get_bundle():
    return load_model()


def scores_chart(scores, best: int) -> go.Figure:
    colors = ["#2E86DE" if d == best else "#B0B8C4" for d in range(10)]
    fig = go.Figure(
        go.Bar(
            x=[str(d) for d in range(10)],
            y=[round(float(s) * 100, 1) for s in scores],
            marker_color=colors,
            text=[f"{s * 100:.1f} %" for s in scores],
            textposition="outside",
            hovertemplate="Chiffre %{x} : %{y:.1f} %<extra></extra>",
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Chiffre",
        yaxis=dict(title="Score estimé (%)", range=[0, 112]),
        showlegend=False,
    )
    return fig


try:
    bundle = get_bundle()
except (FileNotFoundError, ValueError) as err:
    st.error(str(err))
    st.stop()
model, metrics = bundle["model"], bundle["metrics"]

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0
if "result" not in st.session_state:
    st.session_state.result = None

# ----- Header -----
st.title("✏️ DrawAI")
st.subheader("Dessinez un chiffre, laissez l'intelligence artificielle le reconnaître.")

left, right = st.columns([1, 1.2], gap="large")

# ----- Drawing area -----
with left:
    st.markdown("### 1. Dessinez")
    stroke = st.slider("Épaisseur du trait", 8, 40, 20)
    canvas = st_canvas(
        fill_color="#000000",
        stroke_width=stroke,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        update_streamlit=True,
        display_toolbar=False,
        key=f"canvas_{st.session_state.canvas_key}",
    )
    b1, b2 = st.columns(2)
    predict_clicked = b1.button("Prédire", type="primary", use_container_width=True)
    if b2.button("Effacer", use_container_width=True):
        st.session_state.canvas_key += 1
        st.session_state.result = None
        st.rerun()

if predict_clicked:
    image = canvas.image_data
    if is_blank(image):
        st.session_state.result = ("empty", None)
    else:
        st.session_state.result = ("ok", predict(model, image))

# ----- Results -----
with right:
    st.markdown("### 2. Résultat")
    result = st.session_state.result
    if result is None:
        st.info("Dessinez un chiffre entre 0 et 9 puis cliquez sur « Prédire ».")
    elif result[0] == "empty":
        st.warning("Le canvas est vide : dessinez d'abord un chiffre.")
    else:
        pred = result[1]
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(
                f'<div class="digit-box"><div class="digit">{pred.digit}</div>'
                '<div class="caption">Chiffre reconnu</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.metric("Score de confiance estimé", f"{pred.confidence * 100:.1f} %")
            st.caption(
                "Ce score est une estimation calibrée sur les données d'entraînement : "
                "ce n'est pas une garantie que la réponse soit correcte."
            )
            st.markdown("Image 8 × 8 transmise au modèle :")
            st.image(pred.processed / 16.0, width=96, clamp=True)
        if pred.is_uncertain:
            st.warning(
                "Dessin difficile à reconnaître : le modèle hésite. "
                "Essayez un trait plus net, un chiffre plus grand ou plus simple."
            )
        st.plotly_chart(scores_chart(pred.scores, pred.digit), use_container_width=True)

st.divider()

# ----- Educational section -----
st.markdown("## Comment ça marche ?")
with st.expander("Qu'est-ce que le Machine Learning ?"):
    st.write(
        "Le Machine Learning (apprentissage automatique) consiste à laisser un programme "
        "découvrir lui-même des règles à partir d'exemples, au lieu de les écrire à la main. "
        "On lui montre beaucoup d'images de chiffres avec leur bonne réponse, et il ajuste "
        "ses paramètres pour se tromper de moins en moins."
    )
with st.expander("Comment le modèle apprend-il à reconnaître les chiffres ?"):
    st.write(
        "Chaque image est transformée en 64 nombres (un par pixel). Le modèle compare ces "
        "nombres à ceux des exemples connus et apprend quelles combinaisons de pixels "
        "correspondent à chaque chiffre. Il est ensuite testé sur des images qu'il n'a "
        "jamais vues, pour mesurer honnêtement sa qualité."
    )
with st.expander("Quel jeu de données est utilisé ?"):
    st.write(
        f"Le jeu **Digits** de Scikit-learn : {metrics['n_samples_total']} images de chiffres "
        "manuscrits en 8 × 8 pixels (intensités de 0 à 16). L'entraînement est enrichi de "
        "versions « dessinées sur un canvas » (tailles, positions, épaisseurs et inclinaisons "
        "variées) obtenues à partir des images d'entraînement uniquement."
    )
with st.expander("Comment fonctionne le prétraitement ?"):
    st.markdown(
        "Votre dessin 280 × 280 est converti pour ressembler aux images d'entraînement :\n"
        "1. conversion en niveaux de gris (fond noir, trait clair) ;\n"
        "2. découpage autour du tracé ;\n"
        "3. ajout de marges pour garder les proportions (carré) ;\n"
        "4. réduction à 8 × 8 pixels ;\n"
        "5. centrage du chiffre et mise à l'échelle des intensités de 0 à 16."
    )
with st.expander("Quelles sont les limites de la reconnaissance ?"):
    st.write(
        "L'image 8 × 8 contient peu de détails : des chiffres proches (1 et 7, 4 et 9, 3 et 5, "
        "8 et 9) peuvent être confondus. Le modèle ne connaît que les chiffres de 0 à 9, "
        "isolés ; une lettre ou un symbole recevra tout de même une réponse. Les styles "
        "d'écriture très éloignés des exemples sont moins bien reconnus."
    )

# ----- Technical information -----
st.markdown("## Informations techniques")
t1, t2, t3, t4 = st.columns(4)
t1.metric("Modèle", metrics["model_name"])
t2.metric("Accuracy (jeu de test)", f"{metrics['test_accuracy'] * 100:.1f} %")
t3.metric("Images du jeu de données", metrics["n_samples_total"])
t4.metric("Images de test", metrics["n_test_images"])
st.caption(
    f"Accuracy sur dessins simulés de type canvas (images de test uniquement) : "
    f"{metrics['test_accuracy_canvas_simulated'] * 100:.1f} %. "
    "Technologies : Python, Streamlit, Scikit-learn, NumPy, Pillow, Plotly."
)
