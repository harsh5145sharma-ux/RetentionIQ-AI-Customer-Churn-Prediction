"""
============================================================
   RETENTIONIQ AI CUSTOMER CHURN PREDICTION PROJECT
  Dataset : RetentionIQ Churn Dataset (1,000 customers)
  Author  : HARSH KUMAR SHARMA , PRIYA RANI
  Goal    : Predict which customers are likely to churn
            using machine learning, and extract actionable
            business insights.
============================================================

WORKFLOW
--------
1. Load & Explore (EDA)
2. Data Preprocessing
3. Feature Engineering
4. Model Training (4 classifiers)
5. Evaluation (Accuracy, AUC, Cross-Validation)
6. Best Model Deep-Dive (Feature Importance, Confusion Matrix)
7. Business Insights (Risk Segments, Lift Chart)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

# ─── Design Tokens ───────────────────────────────────────────────
PRIMARY   = '#4F46E5'   # indigo  – primary model / brand colour
SECONDARY = '#10B981'   # emerald – retained customers
DANGER    = '#EF4444'   # red     – churned customers
WARN      = '#F59E0B'   # amber   – medium risk
LIGHT_BG  = '#F8FAFC'
DARK_TEXT = '#1E293B'

plt.rcParams.update({
    'figure.facecolor': LIGHT_BG, 'axes.facecolor': 'white',
    'axes.edgecolor': '#CBD5E1', 'axes.labelcolor': DARK_TEXT,
    'xtick.color': DARK_TEXT, 'ytick.color': DARK_TEXT,
    'text.color': DARK_TEXT, 'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.color': '#CBD5E1',
})


# ════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD & EXPLORE DATA
# ════════════════════════════════════════════════════════════════

df = pd.read_csv(r"C:\Users\hp\Downloads\retention_iq_churn_dataset.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape       : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Target      : 'churned'  (1 = churned, 0 = retained)")
print(f"Churn rate  : {df['churned'].mean()*100:.1f}%")
print(f"Missing vals: {df.isnull().sum().sum()}")
print()
print(df.head(3).to_string())
print()
print("Data types:")
print(df.dtypes)
print()
print("Descriptive statistics:")
print(df.describe().round(2))


# ════════════════════════════════════════════════════════════════
#  STEP 2 — EDA DASHBOARD  (saved → 01_eda_dashboard.png)
# ════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 14), facecolor=LIGHT_BG)
fig.suptitle('Customer Churn — Exploratory Data Analysis',
             fontsize=20, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

# Panel 1 – Churn distribution (bar)
ax0 = fig.add_subplot(gs[0, 0])
counts = df['churned'].value_counts()
bars = ax0.bar(['Retained', 'Churned'], counts.values,
               color=[SECONDARY, DANGER], width=0.5, edgecolor='white')
ax0.set_title('Churn Distribution', fontweight='bold')
ax0.set_ylabel('Count')
for bar, val in zip(bars, counts.values):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
             f'{val}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=10, fontweight='bold')
ax0.set_ylim(0, max(counts.values) * 1.2)

# Panel 2 – Age distribution overlay
ax1 = fig.add_subplot(gs[0, 1])
df[df['churned']==0]['age'].plot(kind='hist', bins=20, ax=ax1, alpha=0.6,
                                  color=SECONDARY, label='Retained', edgecolor='white')
df[df['churned']==1]['age'].plot(kind='hist', bins=20, ax=ax1, alpha=0.6,
                                  color=DANGER, label='Churned', edgecolor='white')
ax1.set_title('Age Distribution by Churn', fontweight='bold')
ax1.set_xlabel('Age'); ax1.set_ylabel('Count'); ax1.legend()

# Panel 3 – Tenure boxplot
ax2 = fig.add_subplot(gs[0, 2])
df_box = df[['tenure_months', 'churned']].copy()
df_box['Label'] = df_box['churned'].map({0: 'Retained', 1: 'Churned'})
sns.boxplot(data=df_box, x='Label', y='tenure_months', ax=ax2,
            palette={'Retained': SECONDARY, 'Churned': DANGER})
ax2.set_title('Tenure Months by Churn', fontweight='bold')
ax2.set_xlabel(''); ax2.set_ylabel('Tenure (months)')

# Panel 4 – Churn rate by contract type
ax3 = fig.add_subplot(gs[1, 0])
ct = df.groupby('contract_type')['churned'].mean().sort_values(ascending=False)
bar_colors = [DANGER if v > 0.5 else WARN if v > 0.3 else SECONDARY for v in ct.values]
bars = ax3.barh(ct.index, ct.values * 100, color=bar_colors, edgecolor='white')
ax3.set_title('Churn Rate by Contract Type', fontweight='bold')
ax3.set_xlabel('Churn Rate (%)')
for bar, val in zip(bars, ct.values):
    ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val*100:.1f}%', va='center', fontsize=10)

# Panel 5 – Monthly charges boxplot
ax4 = fig.add_subplot(gs[1, 1])
df_mc = df[['monthly_charges', 'churned']].copy()
df_mc['Label'] = df_mc['churned'].map({0: 'Retained', 1: 'Churned'})
sns.boxplot(data=df_mc, x='Label', y='monthly_charges', ax=ax4,
            palette={'Retained': SECONDARY, 'Churned': DANGER})
ax4.set_title('Monthly Charges by Churn', fontweight='bold')
ax4.set_xlabel(''); ax4.set_ylabel('Monthly Charges ($)')

# Panel 6 – Payment failures vs churn rate
ax5 = fig.add_subplot(gs[1, 2])
pf = df.groupby('payment_failures')['churned'].mean()
ax5.bar(pf.index, pf.values * 100, color=PRIMARY, edgecolor='white', alpha=0.85)
ax5.set_title('Churn Rate by Payment Failures', fontweight='bold')
ax5.set_xlabel('Number of Payment Failures'); ax5.set_ylabel('Churn Rate (%)')

# Panel 7 – NPS score distribution
ax6 = fig.add_subplot(gs[2, 0])
df[df['churned']==0]['nps_score'].plot(kind='hist', bins=20, ax=ax6, alpha=0.6,
                                        color=SECONDARY, label='Retained', edgecolor='white')
df[df['churned']==1]['nps_score'].plot(kind='hist', bins=20, ax=ax6, alpha=0.6,
                                        color=DANGER, label='Churned', edgecolor='white')
ax6.set_title('NPS Score Distribution', fontweight='bold')
ax6.set_xlabel('NPS Score'); ax6.legend()

# Panel 8 – Churn rate by plan
ax7 = fig.add_subplot(gs[2, 1])
plan_churn = df.groupby('plan')['churned'].mean().sort_values(ascending=False)
ax7.bar(plan_churn.index, plan_churn.values * 100, color=WARN, edgecolor='white', alpha=0.85)
ax7.set_title('Churn Rate by Plan', fontweight='bold')
ax7.set_xlabel('Plan'); ax7.set_ylabel('Churn Rate (%)')
plt.setp(ax7.get_xticklabels(), rotation=20, ha='right', fontsize=9)

# Panel 9 – Logins last 30 days
ax8 = fig.add_subplot(gs[2, 2])
df_ll = df[['logins_last_30d', 'churned']].copy()
df_ll['Label'] = df_ll['churned'].map({0: 'Retained', 1: 'Churned'})
sns.boxplot(data=df_ll, x='Label', y='logins_last_30d', ax=ax8,
            palette={'Retained': SECONDARY, 'Churned': DANGER})
ax8.set_title('Logins (Last 30d) by Churn', fontweight='bold')
ax8.set_xlabel(''); ax8.set_ylabel('Logins')

plt.savefig('01_eda_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[Saved] 01_eda_dashboard.png")


# ════════════════════════════════════════════════════════════════
#  STEP 3 — DATA PREPROCESSING
# ════════════════════════════════════════════════════════════════

# Feature selection – drop raw ID and leaky churn_probability column
features = [
    'age', 'tenure_months', 'monthly_charges', 'total_charges',
    'logins_last_30d', 'features_used', 'support_tickets', 'nps_score',
    'last_login_days_ago', 'email_open_rate', 'payment_failures',
    'plan', 'contract_type', 'payment_method', 'gender', 'region'
]
X = df[features].copy()
y = df['churned']

# Label-encode categorical columns
cat_cols = ['plan', 'contract_type', 'payment_method', 'gender', 'region']
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))

# Impute missing values with column median
imputer = SimpleImputer(strategy='median')
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train / test split  (80% train, 20% test, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (needed for Logistic Regression)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("\n[Preprocessing complete]")
print(f"  Train set : {X_train.shape[0]} rows")
print(f"  Test set  : {X_test.shape[0]} rows")


# ════════════════════════════════════════════════════════════════
#  STEP 4 — MODEL TRAINING & EVALUATION
# ════════════════════════════════════════════════════════════════

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=6, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
print("\n" + "=" * 60)
print("MODEL TRAINING RESULTS")
print("=" * 60)
print(f"{'Model':<22} {'Accuracy':>10} {'Test AUC':>10} {'CV AUC':>10}")
print("-" * 60)

for name, model in models.items():
    # Logistic Regression needs scaled input; tree-based models don't
    Xtr = X_train_sc if name == 'Logistic Regression' else X_train
    Xte = X_test_sc  if name == 'Logistic Regression' else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    cv   = cross_val_score(model, Xtr, y_train, cv=5, scoring='roc_auc').mean()

    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
        'acc': acc, 'auc': auc, 'cv_auc': cv, 'Xte': Xte
    }
    print(f"{name:<22} {acc:>10.3f} {auc:>10.3f} {cv:>10.3f}")

print("-" * 60)


# ════════════════════════════════════════════════════════════════
#  STEP 5 — MODEL COMPARISON CHART  (saved → 02_model_comparison.png)
# ════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=LIGHT_BG)
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

# Grouped bar chart
names   = list(results.keys())
accs    = [results[n]['acc']    for n in names]
aucs    = [results[n]['auc']    for n in names]
cv_aucs = [results[n]['cv_auc'] for n in names]
x = np.arange(len(names)); w = 0.25

axes[0].bar(x - w, accs,    w, label='Accuracy',  color=PRIMARY,   alpha=0.85, edgecolor='white')
axes[0].bar(x,     aucs,    w, label='Test AUC',  color=SECONDARY, alpha=0.85, edgecolor='white')
axes[0].bar(x + w, cv_aucs, w, label='CV AUC',    color=WARN,      alpha=0.85, edgecolor='white')
axes[0].set_xticks(x)
axes[0].set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=10)
axes[0].set_ylim(0.5, 1.05)
axes[0].set_ylabel('Score')
axes[0].set_title('Accuracy & AUC by Model', fontweight='bold')
axes[0].legend()
for bar in axes[0].patches:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{bar.get_height():.2f}', ha='center', fontsize=8)

# ROC curves
colors_roc = [PRIMARY, WARN, SECONDARY, DANGER]
for (name, res), col in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", color=col, lw=2)
axes[1].plot([0, 1], [0, 1], '--', color='#94A3B8', lw=1.5, label='Random')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curves — All Models', fontweight='bold')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig('02_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[Saved] 02_model_comparison.png")


# ════════════════════════════════════════════════════════════════
#  STEP 6 — BEST MODEL DEEP DIVE  (saved → 03_best_model_deepdive.png)
# ════════════════════════════════════════════════════════════════

best_name = max(results, key=lambda n: results[n]['auc'])
best = results[best_name]
print(f"\nBest model: {best_name}  |  AUC = {best['auc']:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, best['y_pred'], target_names=['Retained', 'Churned']))

fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=LIGHT_BG)
fig.suptitle(f'Best Model Deep Dive — {best_name}', fontsize=16, fontweight='bold')

# Confusion matrix
cm = confusion_matrix(y_test, best['y_pred'])
ConfusionMatrixDisplay(cm, display_labels=['Retained', 'Churned']).plot(
    ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title('Confusion Matrix', fontweight='bold')

# Feature importances from Random Forest (tree-native)
rf = results['Random Forest']['model']
feat_imp = pd.Series(rf.feature_importances_, index=X_imp.columns).nlargest(12)
feat_imp.sort_values().plot(kind='barh', ax=axes[1], color=PRIMARY, alpha=0.85, edgecolor='white')
axes[1].set_title('Top 12 Feature Importances\n(Random Forest)', fontweight='bold')
axes[1].set_xlabel('Importance Score')

# Predicted probability distribution
axes[2].hist(best['y_prob'][y_test.values == 0], bins=25, alpha=0.6,
             color=SECONDARY, label='Retained', edgecolor='white')
axes[2].hist(best['y_prob'][y_test.values == 1], bins=25, alpha=0.6,
             color=DANGER, label='Churned', edgecolor='white')
axes[2].axvline(0.5, color=DARK_TEXT, linestyle='--', lw=1.5, label='Threshold = 0.5')
axes[2].set_title('Predicted Probability Distribution', fontweight='bold')
axes[2].set_xlabel('Churn Probability'); axes[2].set_ylabel('Count')
axes[2].legend()

plt.tight_layout()
plt.savefig('03_best_model_deepdive.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] 03_best_model_deepdive.png")


# ════════════════════════════════════════════════════════════════
#  STEP 7 — BUSINESS INSIGHTS  (saved → 04_business_insights.png)
# ════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=LIGHT_BG)
fig.suptitle('Business Insights & Risk Segmentation', fontsize=16, fontweight='bold')

# Risk segmentation (Low / Medium / High)
df_test = X_test.copy()
df_test['churn_prob'] = best['y_prob']
df_test['risk'] = pd.cut(
    best['y_prob'],
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)
risk_counts = df_test['risk'].value_counts().reindex(['Low Risk', 'Medium Risk', 'High Risk'])
axes[0].bar(risk_counts.index, risk_counts.values,
            color=[SECONDARY, WARN, DANGER], edgecolor='white', width=0.5)
axes[0].set_title('Customer Risk Segmentation', fontweight='bold')
axes[0].set_ylabel('Number of Customers')
for bar, val in zip(axes[0].patches, risk_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(val), ha='center', fontweight='bold')

# Cumulative Gains / Lift chart
# → Shows how many churners we'd catch if we targeted top-N% of customers
sorted_idx    = np.argsort(-best['y_prob'])
sorted_actual = y_test.values[sorted_idx]
cum_churn     = np.cumsum(sorted_actual) / sorted_actual.sum()
pct_customers = np.arange(1, len(sorted_actual) + 1) / len(sorted_actual)

axes[1].plot(pct_customers * 100, cum_churn * 100, color=PRIMARY, lw=2.5, label='Model')
axes[1].plot([0, 100], [0, 100], '--', color='#94A3B8', lw=1.5, label='Random Baseline')
axes[1].fill_between(pct_customers * 100, cum_churn * 100, pct_customers * 100,
                     alpha=0.10, color=PRIMARY)
idx_30 = int(0.30 * len(sorted_actual))
axes[1].axvline(30, color=WARN, linestyle=':', lw=1.5)
axes[1].axhline(cum_churn[idx_30] * 100, color=WARN, linestyle=':', lw=1.5)
axes[1].annotate(
    f'Top 30% → {cum_churn[idx_30]*100:.0f}% churners captured',
    xy=(30, cum_churn[idx_30] * 100),
    xytext=(38, cum_churn[idx_30] * 100 - 12),
    arrowprops=dict(arrowstyle='->', color=WARN),
    color=WARN, fontsize=9
)
axes[1].set_xlabel('% Customers Targeted')
axes[1].set_ylabel('% Churners Captured')
axes[1].set_title('Cumulative Gains (Lift) Chart', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('04_business_insights.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] 04_business_insights.png")

print("\n" + "=" * 60)
print("PROJECT COMPLETE  DONE")
print("=" * 60)
print("Output files:")
print("  01_eda_dashboard.png       — 9-panel EDA visualisation")
print("  02_model_comparison.png    — AUC bar chart + ROC curves")
print("  03_best_model_deepdive.png — Confusion matrix, feature importance")
print("  04_business_insights.png   — Risk segments + Lift chart")
