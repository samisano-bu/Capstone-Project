"""Milestone Three figures (make_m3_figures.py). Writes PNGs to ./figs and a fig_log.json of computed values.

Breast cancer and heart figures are regenerated from the raw data with the exact
model specifications in the Milestone One/Two notebooks (verified to reproduce the
reported metrics). Registry and tuning-grid figures are rebuilt from the numeric
outputs printed in the Milestone Two notebooks (raw registry data not used here).
"""
import json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, accuracy_score,
                             log_loss, adjusted_rand_score, recall_score)

warnings.filterwarnings("ignore")
RS = 42
OUT = "figs/"
import os; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8, "axes.grid": True, "grid.alpha": 0.3,
    "savefig.dpi": 300, "figure.dpi": 100,
})
BLUE, RED, GREEN, ORANGE, PURPLE, GREY = "tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple", "0.45"
log = {}

def save(fig, name):
    fig.savefig(OUT + name, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

# ---------------------------------------------------------------- data
bc = load_breast_cancer()
Xbc = pd.DataFrame(bc.data, columns=bc.feature_names)
y_mal = 1 - bc.target                      # 1 = malignant
try:  # same source as the Milestone notebooks
    from sklearn.datasets import fetch_openml
    heart = fetch_openml(name="heart-disease", version=1, as_frame=True, parser="auto").frame
    heart = heart.apply(pd.to_numeric, errors="coerce").dropna().astype(float).reset_index(drop=True)
except Exception:  # offline fallback: a local copy of the same 303-row file
    heart = pd.read_csv("heart.csv", encoding="utf-8-sig").astype(float)
Xh = heart.drop(columns=["target"])
y_t = heart["target"].astype(int).values   # as distributed (1 = no disease, see report)
y_dis = 1 - y_t                            # 1 = disease

# ---------------------------------------------------------------- Figure 1
sd = Xbc.std().sort_values()
log["sd_ratio"] = float(sd.max() / sd.min())
sel = pd.concat([sd.head(5), sd.tail(5)])
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.7), gridspec_kw={"width_ratios": [1, 1.05]})
cols = [BLUE] * 5 + [RED] * 5
ax[0].barh(range(10), sel.values, color=cols)
ax[0].set_yticks(range(10)); ax[0].set_yticklabels(sel.index, fontsize=8)
ax[0].set_xscale("log"); ax[0].set_xlabel("Standard deviation (log scale)")
ax[0].axhline(4.5, color=GREY, lw=0.8, ls=":")
ax[0].set_title("(a) Five smallest and five largest SDs")
ax[0].text(0.97, 0.04, f"max/min SD ratio\n= {log['sd_ratio']:,.0f}", transform=ax[0].transAxes,
           ha="right", va="bottom", fontsize=8, bbox=dict(fc="white", ec=GREY, lw=0.6))
mean_cols = [c for c in Xbc.columns if c.startswith("mean")]
corr = Xbc[mean_cols].corr()
im = ax[1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
lab = [c.replace("mean ", "").replace("fractal dimension", "fractal dim.") for c in mean_cols]
ax[1].set_xticks(range(10)); ax[1].set_xticklabels(lab, rotation=60, ha="right", fontsize=7.5)
ax[1].set_yticks(range(10)); ax[1].set_yticklabels(lab, fontsize=7.5)
ax[1].grid(False); ax[1].set_title("(b) Correlation, ten 'mean' features")
fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03)
fig.tight_layout(w_pad=1.0)
save(fig, "fig1_bc_scale_corr.png")
log["r_radius_perimeter"] = float(corr.loc["mean radius", "mean perimeter"])
log["r_radius_area"] = float(corr.loc["mean radius", "mean area"])
log["n_corr_ge_09"] = int(((corr.abs() >= 0.9).values.sum() - 10) / 2)

# ---------------------------------------------------------------- Figure 2
Zb_s = StandardScaler().fit_transform(Xbc)
pb = PCA(2).fit(Zb_s); Zb = pb.transform(Zb_s)
Zh_s = StandardScaler().fit_transform(Xh)
ph = PCA(2).fit(Zh_s); Zh = ph.transform(Zh_s)
log["bc_pc_var"] = pb.explained_variance_ratio_.round(3).tolist()
log["heart_pc_var"] = ph.explained_variance_ratio_.round(3).tolist()
fig, ax = plt.subplots(2, 2, figsize=(6.5, 4.5))
a = ax[0, 0]
a.scatter(Zb[y_mal == 0, 0], Zb[y_mal == 0, 1], s=7, alpha=0.6, c=BLUE, label="Benign (n=357)")
a.scatter(Zb[y_mal == 1, 0], Zb[y_mal == 1, 1], s=7, alpha=0.6, c=RED, label="Malignant (n=212)")
a.set_xlabel(f"PC1 ({pb.explained_variance_ratio_[0]*100:.1f}%)"); a.set_ylabel(f"PC2 ({pb.explained_variance_ratio_[1]*100:.1f}%)")
a.set_title("(a) Breast cancer, standardized PCA"); a.legend(loc="upper right", markerscale=1.5)
a = ax[0, 1]
a.scatter(Zh[y_dis == 0, 0], Zh[y_dis == 0, 1], s=8, alpha=0.6, c=BLUE, label="No disease (n=165)")
a.scatter(Zh[y_dis == 1, 0], Zh[y_dis == 1, 1], s=8, alpha=0.6, c=RED, label="Disease (n=138)")
a.set_xlabel(f"PC1 ({ph.explained_variance_ratio_[0]*100:.1f}%)"); a.set_ylabel(f"PC2 ({ph.explained_variance_ratio_[1]*100:.1f}%)")
a.set_title("(b) Heart disease, standardized PCA"); a.legend(loc="upper right", markerscale=1.5)
a = ax[1, 0]
names = ["Breast cancer:\nmalignant", "Heart:\ndisease present", "Registry, labeled:\ndeceased", "Registry, clean:\ndeceased"]
rates = [y_mal.mean(), y_dis.mean(), 0.5912, 0.3948]
ns = ["n = 569", "n = 303", "n = 209,442", "n = 141,452"]
bars = a.barh(range(4)[::-1], rates, color=[RED, RED, GREY, GREEN])
for i, (r, n) in enumerate(zip(rates, ns)):
    a.text(r + 0.01, 3 - i, f"{r*100:.1f}%  ({n})", va="center", fontsize=8)
a.set_yticks(range(4)[::-1]); a.set_yticklabels(names, fontsize=8)
a.set_xlim(0, 1.32); a.set_xticks(np.arange(0, 1.01, 0.2)); a.set_xlabel("Share of records in positive class")
a.set_title("(c) Positive-class share")
a = ax[1, 1]
r = heart["age"].corr(heart["chol"]); log["r_age_chol"] = float(r)
a.scatter(heart["age"], heart["chol"], s=8, alpha=0.55, c=PURPLE)
m, b0 = np.polyfit(heart["age"], heart["chol"], 1)
xx = np.linspace(heart.age.min(), heart.age.max(), 10); a.plot(xx, m * xx + b0, c=RED, lw=1.4)
a.set_xlabel("Age (years)"); a.set_ylabel("Serum cholesterol (mg/dl)")
a.set_title(f"(d) Cholesterol vs age (r = {r:.2f})")
fig.tight_layout(h_pad=1.2, w_pad=1.0)
save(fig, "fig2_geometry_balance.png")

# ---------------------------------------------------------------- Figure 3 (registry, from printed outputs)
fig = plt.figure(figsize=(6.5, 4.3))
gs = fig.add_gridspec(2, 1, height_ratios=[0.9, 1.6], hspace=0.35)
a = fig.add_axes([0.0, 0.66, 1.0, 0.30]); a.axis("off"); a.set_xlim(0, 1); a.set_ylim(0, 1)
a.text(0.0, 1.05, "(a) Registry cohort construction", fontsize=9.5, ha="left", va="bottom")
boxes = [
    (0.0, "1,778,176 records\n38 fields, 2000–2019\n(registry as distributed)"),
    (0.35, "209,442 labeled records\nvital status recorded\n(88.2% of records lack it)"),
    (0.70, "141,452 clean cohort\nmodeling population\ndeath rate 39.5%"),
]
for x, t in boxes:
    a.add_patch(FancyBboxPatch((x + 0.005, 0.38), 0.29, 0.55, boxstyle="round,pad=0.01", fc="#eef3fb", ec=BLUE, lw=1))
    a.text(x + 0.15, 0.655, t, ha="center", va="center", fontsize=8)
for x in (0.30, 0.65):
    a.annotate("", xy=(x + 0.045, 0.655), xytext=(x - 0.005, 0.655), arrowprops=dict(arrowstyle="-|>", color="0.2", lw=1.1))
a.text(0.5, 0.12, "Excluded: 65,665 death-certificate-only (SDO) cases and the 19 registries with no outcome variation "
       "(67,990 records in total;\ndeath rate falls from 59.1% to 39.5%)", ha="center", va="center", fontsize=7.6, color=RED)
a2 = fig.add_axes([0.33, 0.07, 0.62, 0.48])
dm = pd.DataFrame([
    ("SDO: death certificate only", 65665, 1.0000),
    ("Clinical investigation (Pesquisa)", 3227, 0.9092),
    ("Missing", 784, 0.8699),
    ("Histology of metastasis", 1733, 0.8332),
    ("Clinical only (Clínico)", 1213, 0.7890),
    ("Tumor markers", 176, 0.7273),
    ("Cytology", 1248, 0.4559),
    ("Histology of primary tumor", 135396, 0.3799),
], columns=["means", "n", "rate"])
yy = np.arange(len(dm))[::-1]
a2.barh(yy, dm.rate, color=[RED] + [BLUE] * 7)
for y_, (_, row) in zip(yy, dm.iterrows()):
    a2.text(row.rate + 0.01, y_, f"{row.rate:.3f}  (n = {row.n:,})", va="center", fontsize=7.8)
a2.set_yticks(yy); a2.set_yticklabels(dm.means, fontsize=8)
a2.axvline(0.5912, color=GREY, ls="--", lw=1); a2.text(0.60, -0.95, "labeled-cohort mean 0.591", fontsize=7.5, color=GREY, ha="left")
a2.set_xlim(0, 1.45); a2.set_ylim(-1.3, len(dm) - 0.4); a2.set_xticks(np.arange(0, 1.01, 0.2))
a2.set_xlabel("Observed mortality (labeled cohort, 209,442 records)")
a2.text(-0.52, 1.03, "(b) Mortality by basis of diagnosis (Diagnostic.means)", transform=a2.transAxes, fontsize=9.5, ha="left", va="bottom")
save(fig, "fig3_registry.png")

# ---------------------------------------------------------------- split A (Milestone One) breast cancer models
t = bc.target  # 0 = malignant (Milestone One coding)
XA_tr, XA_te, yA_tr, yA_te = train_test_split(bc.data, t, test_size=0.25, random_state=RS, stratify=t)
cv5 = StratifiedKFold(5, shuffle=True, random_state=RS)
lr_g = GridSearchCV(Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))]),
                    {"lr__C": np.logspace(-3, 2, 12)}, scoring="roc_auc", cv=cv5, n_jobs=-1).fit(XA_tr, yA_tr)
svm_g = GridSearchCV(Pipeline([("s", StandardScaler()), ("svc", SVC(kernel="rbf", probability=True))]),
                     {"svc__C": np.logspace(-2, 2, 6), "svc__gamma": np.logspace(-4, 0, 6)},
                     scoring="roc_auc", cv=cv5, n_jobs=-1).fit(XA_tr, yA_tr)
svl_g = GridSearchCV(Pipeline([("s", StandardScaler()), ("svc", SVC(kernel="linear", probability=True))]),
                     {"svc__C": np.logspace(-2, 2, 8)}, scoring="roc_auc", cv=cv5, n_jobs=-1).fit(XA_tr, yA_tr)
tr_g = GridSearchCV(DecisionTreeClassifier(random_state=RS), {"max_depth": [2, 3, 4, 5, 6, 8, None],
                    "min_samples_leaf": [1, 3, 5, 10]}, scoring="roc_auc", cv=cv5, n_jobs=-1).fit(XA_tr, yA_tr)
rf_g = GridSearchCV(RandomForestClassifier(random_state=RS, n_jobs=-1), {"n_estimators": [100, 300],
                    "max_depth": [None, 6, 10], "max_features": ["sqrt", 0.5]}, scoring="roc_auc", cv=cv5, n_jobs=-1).fit(XA_tr, yA_tr)
modelsA = {"Logistic regression (L2)": lr_g, "SVM, RBF kernel": svm_g, "SVM, linear kernel": svl_g,
           "Random forest": rf_g, "Decision tree (depth 5)": tr_g}
log["splitA"] = {}
for k, g in modelsA.items():
    p_mal = g.predict_proba(XA_te)[:, 0]          # P(malignant)
    pred_mal = (g.predict(XA_te) == 0).astype(int)
    yte_mal = (yA_te == 0).astype(int)
    cm = confusion_matrix(yte_mal, pred_mal, labels=[1, 0])
    log["splitA"][k] = dict(auc=round(roc_auc_score(yte_mal, p_mal), 4), acc=round(accuracy_score(yte_mal, pred_mal), 4),
                            sens=round(recall_score(yte_mal, pred_mal), 4), cm=cm.tolist(),
                            params={kk: (float(v) if isinstance(v, (float, np.floating)) else v) for kk, v in g.best_params_.items()})

# ---------------------------------------------------------------- Figure 4 (modeling diagnostics)
fig, ax = plt.subplots(2, 2, figsize=(6.5, 5.0))
a = ax[0, 0]
depths = range(1, 16); trn, tst = [], []
for d in depths:
    tm = DecisionTreeClassifier(max_depth=d, random_state=RS).fit(XA_tr, yA_tr)
    trn.append(tm.score(XA_tr, yA_tr)); tst.append(tm.score(XA_te, yA_te))
a.plot(depths, trn, "o-", ms=3, c=RED, label="Training"); a.plot(depths, tst, "s-", ms=3, c=BLUE, label="Held-out test")
a.set_xlabel("Maximum tree depth"); a.set_ylabel("Accuracy"); a.legend(loc="center right")
a.set_title("(a) Single tree: depth vs accuracy (BC)")
log["tree_depth_curve"] = {"train": [round(v, 4) for v in trn], "test": [round(v, 4) for v in tst]}
a = ax[0, 1]
XB_tr, XB_te, yB_tr, yB_te = train_test_split(Xbc, y_mal, test_size=0.25, random_state=RS, stratify=y_mal)
XH_tr, XH_te, yH_tr, yH_te = train_test_split(Xh.values, y_t, test_size=0.25, random_state=RS, stratify=y_t)
staged = {}
for nm, (a_tr, a_te, b_tr, b_te), col in [("BC", (XB_tr, XB_te, yB_tr, yB_te), BLUE), ("Heart", (XH_tr, XH_te, yH_tr, yH_te), RED)]:
    g0 = GradientBoostingClassifier(learning_rate=0.1, max_depth=3, n_estimators=500, random_state=RS).fit(a_tr, b_tr)
    te = [log_loss(b_te, p[:, 1]) for p in g0.staged_predict_proba(a_te)]
    trl = [log_loss(b_tr, p[:, 1]) for p in g0.staged_predict_proba(a_tr)]
    it = np.arange(1, 501); best = int(np.argmin(te)) + 1
    staged[nm] = dict(best=best, min=round(min(te), 4), final=round(te[-1], 4), train_final=float(trl[-1]))
    a.plot(it, te, c=col, lw=1.4, label=f"{nm} test (min at {best})")
    a.plot(it, trl, c=col, lw=1.0, ls="--", label=f"{nm} training")
    a.axvline(best, c=col, lw=0.8, ls=":")
log["staged"] = staged
a.set_xlabel("Boosting iteration"); a.set_ylabel("Log loss (deviance)"); a.legend(fontsize=7, loc="upper right")
a.set_title("(b) GBM staged deviance (BC, heart)")
a = ax[1, 0]
grid = np.array([[0.9615, 0.9781, 0.9861, 0.9894, 0.9902], [0.9823, 0.9893, 0.9898, 0.9923, 0.9922],
                 [0.9903, 0.9920, 0.9927, 0.9927, 0.9927], [0.9869, 0.9910, 0.9921, 0.9922, 0.9922],
                 [0.9847, 0.9876, 0.9880, 0.9880, 0.9880]])
im = a.imshow(grid, cmap="viridis", aspect="auto"); a.grid(False)
a.set_xticks(range(5)); a.set_xticklabels([25, 50, 100, 200, 400]); a.set_yticks(range(5)); a.set_yticklabels([0.01, 0.05, 0.1, 0.3, 1.0])
a.set_xlabel("Number of trees"); a.set_ylabel("Learning rate")
for i in range(5):
    for j in range(5):
        a.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=7, color="white" if grid[i, j] < 0.985 else "black")
a.set_title("(c) GBM tuning grid, CV ROC-AUC (BC)")
fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
a = ax[1, 1]
mets = ["Manhattan (p=1)", "Euclidean (p=2)", "Minkowski (p=3)", "Cosine", "Chebyshev (p=∞)", "Mahalanobis"]
bc_auc = [0.9985, 0.9973, 0.9958, 0.9969, 0.9878, 0.9608]
ht_auc = [0.9094, 0.8481, 0.8502, 0.8547, 0.7324, 0.8564]
yy = np.arange(len(mets))[::-1]
for v1, v2, y_ in zip(bc_auc, ht_auc, yy):
    a.plot([v2, v1], [y_, y_], c="0.75", lw=1.2, zorder=1)
a.scatter(bc_auc, yy, s=34, c=BLUE, label="Breast cancer", zorder=3)
a.scatter(ht_auc, yy, s=34, c=RED, marker="s", label="Heart disease", zorder=3)
a.set_yticks(yy); a.set_yticklabels(mets, fontsize=8); a.set_xlim(0.70, 1.01); a.set_ylim(-0.6, len(mets) - 0.4)
a.set_xlabel("Held-out ROC-AUC (tuned k)"); a.legend(loc="upper left", fontsize=7.5)
a.set_title("(d) KNN: distance metric comparison")
fig.tight_layout(h_pad=1.3, w_pad=1.0)
save(fig, "fig4_modeling.png")

# ---------------------------------------------------------------- Figure 5 (results, BC + heart)
fig, ax = plt.subplots(2, 2, figsize=(6.5, 4.35))
a = ax[0, 0]
yte_mal = (yA_te == 0).astype(int)
for (k, g), col, ls in zip([(k, modelsA[k]) for k in ["Logistic regression (L2)", "SVM, RBF kernel", "Random forest", "Decision tree (depth 5)"]],
                           [BLUE, GREEN, ORANGE, PURPLE], ["-", "--", "-.", ":"]):
    fpr, tpr, _ = roc_curve(yte_mal, g.predict_proba(XA_te)[:, 0])
    a.plot(fpr, tpr, c=col, ls=ls, lw=1.5, label=f"{k.split(' (')[0] if 'tree' not in k else 'Decision tree'} ({log['splitA'][k]['auc']:.4f})")
a.set_xlim(-0.005, 0.30); a.set_ylim(0.70, 1.005)
a.set_xlabel("False positive rate (zoomed to 0–0.3)"); a.set_ylabel("Sensitivity (malignant)")
a.legend(loc="lower right", fontsize=7.2, title="Model (test AUC)", title_fontsize=7.2)
a.set_title("(a) Breast cancer ROC (n = 143)")
a = ax[0, 1]
cm = np.array(log["splitA"]["Logistic regression (L2)"]["cm"])
a.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15); a.grid(False)
for i in range(2):
    for j in range(2):
        a.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13, color="white" if cm[i, j] > 40 else "black")
a.set_xticks([0, 1]); a.set_xticklabels(["Malignant", "Benign"]); a.set_yticks([0, 1]); a.set_yticklabels(["Malignant", "Benign"])
a.set_xlabel("Predicted"); a.set_ylabel("Actual")
a.set_title("(b) Confusion matrix (logistic)")
a = ax[1, 0]
rf_best = rf_g.best_estimator_
imp = pd.Series(rf_best.feature_importances_, index=bc.feature_names).nlargest(10)[::-1]
log["rf_top10"] = {k: round(float(v), 4) for k, v in imp[::-1].items()}
a.barh(range(10), imp.values, color=GREEN)
a.set_yticks(range(10)); a.set_yticklabels(imp.index, fontsize=8); a.set_xlabel("Mean decrease in impurity")
a.set_title("(c) Random forest: top-10 importances")
a = ax[1, 1]
hm = {}
lr_h = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))]).fit(XH_tr, yH_tr)
rf_h = RandomForestClassifier(n_estimators=300, max_depth=5, max_features="sqrt", min_samples_leaf=3, random_state=RS, n_jobs=-1).fit(XH_tr, yH_tr)
gb_h = GradientBoostingClassifier(learning_rate=0.05, max_depth=2, min_samples_leaf=5, n_estimators=100, subsample=0.7, random_state=RS).fit(XH_tr, yH_tr)
kn_h = Pipeline([("s", StandardScaler()), ("k", KNeighborsClassifier(n_neighbors=27, p=1, weights="distance"))]).fit(XH_tr, yH_tr)
yte_dis = 1 - yH_te
for nm, mdl, col, ls in [("KNN, Manhattan k=27", kn_h, BLUE, "-"), ("Random forest", rf_h, ORANGE, "-."),
                         ("Gradient boosting", gb_h, RED, "--"), ("Logistic regression", lr_h, PURPLE, ":")]:
    s_dis = mdl.predict_proba(XH_te)[:, 0]         # P(target = 0) = P(disease)
    auc = roc_auc_score(yte_dis, s_dis); hm[nm] = round(auc, 4)
    fpr, tpr, _ = roc_curve(yte_dis, s_dis)
    a.plot(fpr, tpr, c=col, ls=ls, lw=1.5, label=f"{nm} ({auc:.4f})")
a.plot([0, 1], [0, 1], c=GREY, lw=0.8, ls="--")
a.set_xlabel("False positive rate"); a.set_ylabel("Sensitivity (disease)")
a.legend(loc="lower right", fontsize=7.2, title="Model (test AUC)", title_fontsize=7.2)
a.set_title("(d) Heart disease ROC (n = 76)")
log["heart_auc"] = hm
log["heart_test_n"] = int(len(yH_te)); log["heart_test_disease"] = int(yte_dis.sum())
log["bcA_test_malignant"] = int(yte_mal.sum())
fig.tight_layout(h_pad=1.3, w_pad=1.0)
save(fig, "fig5_results_clinical.png")

# ---------------------------------------------------------------- Figure 6 (registry results, printed outputs)
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.85), gridspec_kw={"width_ratios": [0.9, 1.1]})
a = ax[0]
metrics = ["ROC-AUC", "Accuracy", "Brier"]
cont = [0.9691, 0.9023, 0.0687]; clean = [0.9311, 0.8554, 0.1020]
x = np.arange(3)
a.bar(x - 0.2, cont, 0.38, color=RED, label="Contaminated")
a.bar(x + 0.2, clean, 0.38, color=GREEN, label="Clean")
for i in range(3):
    a.text(x[i] - 0.2, cont[i] + 0.015, f"{cont[i]:.3f}", ha="center", fontsize=6.8, rotation=90, va="bottom")
    a.text(x[i] + 0.2, clean[i] + 0.015, f"{clean[i]:.3f}", ha="center", fontsize=6.8, rotation=90, va="bottom")
a.set_xticks(x); a.set_xticklabels(metrics); a.set_ylim(0, 1.25); a.set_yticks(np.arange(0, 1.01, 0.2))
a.set_title("(a) Held-out performance by cohort")
a = ax[1]
feats = ["Diagnostic.means", "year", "topography_group", "RCBP.Name", "morphology_group", "Age", "Extension",
         "Degree.of.Education", "State.Civil", "Raca.Color", "Gender"]
pi_cont = {"Diagnostic.means": 0.20775, "year": 0.06574, "RCBP.Name": 0.03674, "topography_group": 0.02954, "morphology_group": 0.02953,
           "Age": 0.01198, "Extension": 0.00911, "Degree.of.Education": 0.00550, "State.Civil": 0.00279, "Raca.Color": 0.00174, "Gender": 0.00069}
pi_clean = {"year": 0.13440, "topography_group": 0.06270, "RCBP.Name": 0.05711, "morphology_group": 0.03356, "Age": 0.02666, "Extension": 0.02102,
            "Degree.of.Education": 0.01334, "State.Civil": 0.00507, "Raca.Color": 0.00454, "Gender": 0.00179, "Diagnostic.means": 0.00092}
yy = np.arange(len(feats))[::-1]
a.barh(yy + 0.2, [pi_cont[f] for f in feats], 0.4, color=RED, label="Contaminated")
a.barh(yy - 0.2, [pi_clean[f] for f in feats], 0.4, color=GREEN, label="Clean")
a.set_yticks(yy); a.set_yticklabels(feats, fontsize=7.8)
a.set_xlabel("Drop in test ROC-AUC when shuffled"); a.legend(loc="lower right", fontsize=7.5)
a.set_title("(b) Permutation importance, HistGB")
fig.tight_layout(w_pad=1.2)
save(fig, "fig6_registry_results.png")

# ---------------------------------------------------------------- Figure 7 (unsupervised)
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.85))
a = ax[0]
km = KMeans(2, random_state=RS, n_init=20).fit(Zb_s)
lab = km.labels_
if (lab == y_mal).mean() < 0.5: lab = 1 - lab
mis = lab != y_mal
log["kmeans"] = dict(ari=round(adjusted_rand_score(y_mal, km.labels_), 4), misassigned=int(mis.sum()), agree=round(float(1 - mis.mean()), 4))
a.scatter(Zb[y_mal == 0, 0], Zb[y_mal == 0, 1], s=6, alpha=0.55, c=BLUE, label="Benign (true)")
a.scatter(Zb[y_mal == 1, 0], Zb[y_mal == 1, 1], s=6, alpha=0.55, c=RED, label="Malignant (true)")
a.scatter(Zb[mis, 0], Zb[mis, 1], s=26, facecolors="none", edgecolors="k", lw=0.8, label=f"Misassigned ({mis.sum()})")
a.set_xlabel("PC1"); a.set_ylabel("PC2"); a.legend(loc="lower right", fontsize=7, markerscale=1.2)
a.set_title(f"(a) K-Means, k = 2 vs diagnosis (ARI = {log['kmeans']['ari']:.3f})")
a = ax[1]
link = ["Ward", "Average", "Single", "Complete"]
ari = [0.5750, 0.0073, 0.0048, 0.0048]; sil = [0.3394, 0.6340, 0.6607, 0.6607]; coph = [0.5862, 0.8411, 0.7928, 0.6746]
big = ["68%", "99.5%", "99.6%", "99.6%"]
x = np.arange(4)
a.bar(x - 0.27, ari, 0.27, color=BLUE, label="ARI vs diagnosis")
a.bar(x, sil, 0.27, color=RED, label="Silhouette")
a.bar(x + 0.27, coph, 0.27, color=GREEN, label="Cophenetic corr.")
a.set_xticks(x); a.set_xticklabels([f"{l}\n({b_})" for l, b_ in zip(link, big)]); a.set_ylim(0, 1.3); a.set_yticks(np.arange(0, 1.01, 0.2))
a.set_ylabel("Score"); a.legend(loc="upper left", fontsize=7, ncol=2)
a.set_title("(b) Linkage choice, breast cancer (k = 2)")
fig.tight_layout(w_pad=1.2)
save(fig, "fig7_unsupervised.png")

json.dump(log, open("fig_log.json", "w"), indent=1, default=str)
print(json.dumps(log, indent=1, default=str))
