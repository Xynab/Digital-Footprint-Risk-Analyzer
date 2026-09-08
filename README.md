# Digital Footprint Risk Analyzer

A Streamlit app that scores a user's digital privacy risk from their self-reported online habits (screen time, profile visibility, 2FA, password strength, app permissions, breach exposure) using a trained classifier, blended with a rule-based username-safety check — with per-user login, feature-importance explainability, and risk history tracking.

---

## How it works

**1. Risk model** — three classifiers (Random Forest, Gradient Boosting, Logistic Regression) are trained on a 1,000-row labeled dataset of digital-behavior features; the best performer by accuracy is selected and saved (`models/best_model.pkl`). On the dataset used here, **Logistic Regression** came out on top.

**2. Username heuristic** — as a second, independent signal, the app scores the entered username itself: short length, presence of digits, or matching a common/default name (e.g. "admin", "test") each add to a risk penalty. This is deliberately simple and rule-based, not learned.

**3. Final score** — the ML model's predicted probability of the highest-risk class is combined with the username penalty (capped at 100) to produce the final risk score and HIGH/MEDIUM/LOW label.

**4. Explainability** — the app shows a horizontal bar chart of feature importances (`coef_` for the deployed Logistic Regression model) so a user can see which inputs drove their score. This is coefficient-based importance, not a SHAP/LIME explanation.

**5. Accounts & history** — Streamlit-native login/register, with results logged per-user (`history/results.csv`) and shown back to them as a table.

## Tech stack

Streamlit, scikit-learn, pandas/NumPy, Plotly (gauge + bar charts), bcrypt (password hashing)

## Model performance

~96% overall accuracy on the held-out test split. Worth noting: the dataset's risk classes are imbalanced, and the rarest class had very few test examples, so precision/recall on that class is noticeably weaker than the headline accuracy suggests — a realistic caveat for a small, single-source dataset rather than a production-scale one.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dataset, training notebook (`train_model.ipynb`), evaluation notebook (`evaluate.ipynb`), and explainability notebook (`explainability.ipynb`) are included if you want to reproduce or retrain the model.

## Security note

User passwords are hashed with `bcrypt` before being stored (not plaintext). Login data (`users/`) and per-user history (`history/*.csv`) are excluded from version control via `.gitignore` — they're runtime data, not part of the app itself.
