import gradio as gr
import pandas as pd
import numpy as np
import spaces

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

OPTIONS = ["A", "B", "C", "D", "E"]

# Load Training Data
train = pd.read_csv("train.csv")


def make_text(prompt, a, b, c, d, e):
    return f"{prompt} A: {a} B: {b} C: {c} D: {d} E: {e}"


X = [
    make_text(r["prompt"], r["A"], r["B"], r["C"], r["D"], r["E"])
    for _, r in train.iterrows()
]

y = train["answer"].values

# Train Model
model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                stop_words="english",
            ),
        ),
        ("clf", LogisticRegression(max_iter=1000)),
    ]
)

model.fit(X, y)
print("Model trained on", len(X), "questions")


# ZeroGPU Compatibility
@spaces.GPU
def predict(prompt, a, b, c, d, e):
    if not prompt.strip():
        return "Please enter a question.", None

    text = make_text(prompt, a, b, c, d, e)

    proba = model.predict_proba([text])[0]
    classes = list(model.classes_)

    order = np.argsort(-proba)

    top3 = " ".join(classes[i] for i in order[:3])

    table = [
        [classes[i], round(float(proba[i]), 4)]
        for i in order
    ]

    return f"### Top-3 prediction: `{top3}`", table


EXAMPLE = [
    "What is accelerator-based light-ion fusion?",
    "A process where heavy ions are accelerated into a target to trigger fusion.",
    "A method in which light ions are accelerated to high energy and collide with a target, producing fusion reactions.",
    "A technique that uses lasers alone to compress fuel pellets without any acceleration.",
    "A magnetic confinement scheme that traps plasma using superconducting coils.",
    "A chemical reaction that releases energy without any nuclear process.",
]

# UI
with gr.Blocks(title="Smart MCQ Solver") as demo:

    gr.Markdown(
        """
# Smart MCQ Solver

Enter a question and five options.

The model ranks all options and returns the **Top-3 prediction**
in Kaggle MAP@3 format.

**Model:** TF-IDF + Logistic Regression (MAP@3 = 0.7394)

Created by **Ishank Gupta (23f1002033)**
"""
    )

    with gr.Row():

        with gr.Column():

            prompt = gr.Textbox(
                label="Question / Prompt",
                lines=2,
            )

            a = gr.Textbox(label="Option A", lines=2)
            b = gr.Textbox(label="Option B", lines=2)
            c = gr.Textbox(label="Option C", lines=2)
            d = gr.Textbox(label="Option D", lines=2)
            e = gr.Textbox(label="Option E", lines=2)

            btn = gr.Button("Predict", variant="primary")

        with gr.Column():

            out_text = gr.Markdown()

            out_table = gr.Dataframe(
                headers=["Option", "Probability"],
                label="All options ranked",
            )

    gr.Examples(
        examples=[EXAMPLE],
        inputs=[prompt, a, b, c, d, e],
    )

    btn.click(
        fn=predict,
        inputs=[prompt, a, b, c, d, e],
        outputs=[out_text, out_table],
    )


if __name__ == "__main__":
    demo.launch()