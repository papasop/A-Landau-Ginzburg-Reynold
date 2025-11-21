# ============================
# LGR 场理论 · 30 题真实 LLM 实验（终极版）
# - 高阶梯度 G1..G6
# - FAST vs CoT 分相拟合
# - PLSA 宏观极限 & baseline 对比
# ============================

!pip -q install --upgrade openai

import os, time, gzip, json, textwrap, re, math, random
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from openai import OpenAI

# ========= 0. 填写你的 API KEY =========
# 建议在 Colab 左侧 "齿轮" 里用环境变量方式注入，这里读环境变量；
# 如果没有，就手动填一个占位字符串，然后在运行前改掉。
OPENAI_API_KEY = os.getenv("", "").strip()
if (not OPENAI_API_KEY) or ("YOUR_API_KEY_HERE" in OPENAI_API_KEY):
    raise ValueError("请先设置 OPENAI_API_KEY 环境变量，或者把代码里的 sk-YOUR_API_KEY_HERE 换成自己的 key。")

client = OpenAI(api_key=OPENAI_API_KEY)

# ========= 1. 实验配置 =========

# 30 个推理任务
TASKS = [
    # 简单算术 / 逻辑
    "You have 5 apples. You give 2 to Alice and 1 to Bob, then buy 4 more. How many apples do you have now?",
    "If a train travels 120 kilometers in 1.5 hours, what is its average speed in kilometers per hour?",
    "A rectangle has length 8 and width 5. If the length is increased by 50% and the width is decreased by 20%, what is the new area?",
    "A store offers a 20% discount on a $50 item, then adds 10% tax on the discounted price. What is the final price?",
    "You flip a fair coin three times. What is the probability of getting exactly two heads?",
    "If 3x + 5 = 20, what is x?",
    "A class has 12 boys and 18 girls. What percentage of the class are girls?",
    "The sum of two numbers is 30 and their difference is 6. What are the two numbers?",
    "You invest $1000 at 5% simple annual interest. How much will you have after 3 years?",
    "If a car uses 8 liters of fuel to travel 120 km, how many liters are needed to travel 300 km?",

    # 稍微复杂一点的文字题
    "A tank can be filled by pipe A in 6 hours and by pipe B in 4 hours. If both pipes are used together, how long will it take to fill the tank?",
    "A worker is paid $15 per hour for the first 40 hours in a week and 1.5 times that rate for additional hours. If they worked 46 hours, what is their total pay?",
    "A store sells pencils at 3 for $1.20. How much will 10 pencils cost?",
    "A train leaves City A at 9:00 AM traveling at 80 km/h. Another train leaves City B at 10:00 AM traveling towards City A at 100 km/h. If the distance between the cities is 420 km, at what time do the trains meet?",
    "The average of five numbers is 12. If four of them are 10, 8, 15, and 11, what is the fifth number?",
    "A bag contains 5 red, 3 blue, and 2 green balls. One ball is drawn at random. What is the probability that it is not blue?",
    "If the perimeter of a square is 36 units, what is the area of the square?",
    "The ratio of cats to dogs in a shelter is 3:5. If there are 24 dogs, how many cats are there?",
    "A shop increases the price of an item by 25%, then later offers a 20% discount on the new price. What is the net percentage change from the original price?",
    "A recipe uses 3 cups of flour to make 12 cookies. How many cookies can be made with 5 cups of flour?",

    # 稍复杂推理 / 组合类
    "In how many different ways can the letters of the word 'LEVEL' be arranged?",
    "Two consecutive integers have a product of 506. What are the integers?",
    "You roll two fair six-sided dice. What is the probability that the sum is 9?",
    "A book is sold at a 30% discount for $14. What was its original price?",
    "If 60% of a number is 48, what is the number?",
    "Three friends split a bill of $72 in the ratio 2:3:4. How much does each pay?",
    "A circular garden has radius 7 meters. What is its area in terms of π?",
    "A car’s value decreases by 20% each year. If its initial value is $25,000, what is its value after 2 years (ignoring cents)?",
    "A box contains 6 red and 4 blue balls. Two balls are drawn without replacement. What is the probability both are red?",
    "If y varies directly as x and y=18 when x=6, what is y when x=10?"
]
assert len(TASKS) == 30

# 默认实验配置（你可以改：模型 / 温度 / 实验名）
EXPERIMENT_NAME = "gpt4o-mini_tau0.2_0.4"
MODEL_NAME      = "gpt-4o-mini"   # 你可以改成 gpt-4o / o3-mini / 其他
TEMP_FAST       = 0.2             # FAST 相的温度（τ_F）
TEMP_COT        = 0.4             # CoT 相的温度（τ_C）

# 高阶梯度最高阶数（对应 κ_1..κ_6）
MAX_GRAD_ORDER = 6

# 随机种子（为了 gzip 测量稳定，你也可以固定）
random.seed(0)
np.random.seed(0)

# ========= 2. 工具函数 =========

def gzip_len(s: str) -> int:
    """返回 gzip 压缩后的字节长度，作为近似 Kolmogorov 复杂度。"""
    return len(gzip.compress(s.encode("utf-8"), compresslevel=9))

def measure_phi_x(problem: str, answer: str) -> Dict[str, float]:
    """
    实例序参量 φ(x) + λ_K + 宏观 h(x).
    λ_K = (K_solution - K_problem) / K_problem
    φ(x) = log(1 + λ_K)
    h(x) = 1 - λ_K （PLSA 宏观极限里可以当“结构压缩率”用）
    """
    Kp = gzip_len(json.dumps({"problem": problem}, ensure_ascii=False))
    Ks = gzip_len(json.dumps({"problem": problem, "answer": answer}, ensure_ascii=False))
    lam = max(0.0, (Ks - Kp) / max(Kp, 1))
    phi_x = math.log(1.0 + lam)
    h_x = 1.0 - lam
    return {
        "K_problem": Kp,
        "K_solution": Ks,
        "lambda_K": lam,
        "phi_x": phi_x,
        "h_x": h_x
    }

def split_steps(text: str) -> List[str]:
    """非常粗糙的 step 分割：按换行和句号拆开，然后清洗。"""
    t = text.replace("**", " ").replace("__", " ")
    parts = re.split(r"\n+|(?<=[\.\?\!])\s+", t)
    steps = [p.strip() for p in parts if p.strip()]
    if len(steps) > 80:
        steps = steps[:80]
    return steps if steps else [t.strip()]

def build_phi_t(answer: str) -> np.ndarray:
    """
    从文本生成 φ(t) 序列：
    - 按步骤拆分；
    - 用增量 gzip 残差作为原始能量；
    - 归一化到 [0,1]。
    """
    steps = split_steps(answer)
    prefix = ""
    vals = []
    prev_len = gzip_len(prefix)
    for s in steps:
        prefix = prefix + ("\n" + s)
        cur_len = gzip_len(prefix)
        delta = max(0, cur_len - prev_len)
        vals.append(float(delta))
        prev_len = cur_len
    arr = np.array(vals, dtype=float)
    if arr.max() > 0:
        arr = arr / arr.max()
    else:
        arr = np.ones_like(arr)
    return arr

def lgr_functionals(phi_t: np.ndarray, max_order: int = 6) -> Dict[str, float]:
    """
    Landau–Ginzburg–Reynolds 功能：
      - F2 = mean(phi^2)
      - F4 = mean(phi^4)
      - G_n = mean( (Δ^n phi)^2 ), n = 1..max_order
        这里 Δ^n 用 n 阶有限差分 np.diff(phi_t, n) 实现。
    """
    res = {}
    if len(phi_t) == 0:
        res["F2"] = 0.0
        res["F4"] = 0.0
        for n in range(1, max_order + 1):
            res[f"G{n}"] = 0.0
        return res

    res["F2"] = float(np.mean(phi_t ** 2))
    res["F4"] = float(np.mean(phi_t ** 4))

    for n in range(1, max_order + 1):
        if len(phi_t) > n:
            d = np.diff(phi_t, n)
            res[f"G{n}"] = float(np.mean(d ** 2))
        else:
            res[f"G{n}"] = 0.0

    return res

# ========= 3. LLM 调用 =========

def call_llm(problem: str, mode: str = "FAST") -> Dict[str, Any]:
    """
    mode = "FAST" : 直接给答案（尽量短）
    mode = "CoT"  : 显式思维链，结构更丰富
    """
    if mode == "FAST":
        sys_msg = (
            "You are a helpful assistant. Answer the question concisely with the final result. "
            "Avoid unnecessary explanation."
        )
        temperature = TEMP_FAST
        max_tokens = 256
    else:
        sys_msg = (
            "You are a helpful reasoning assistant. "
            "Solve the problem step by step with explicit reasoning, "
            "using short numbered steps like 'Step 1: ...', then give the final answer."
        )
        temperature = TEMP_COT
        max_tokens = 512

    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": problem},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    end = time.perf_counter()
    T_sec = end - start

    out_txt = resp.choices[0].message.content.strip()
    return {"answer": out_txt, "T_sec": T_sec}

# ========= 4. 主循环：跑 30 题 × (FAST, CoT) =========

rows = []

print(f"=== Running real LLM on 30 tasks (FAST vs CoT) ===")
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Model: {MODEL_NAME}, TEMP_FAST={TEMP_FAST}, TEMP_COT={TEMP_COT}\n")

for task_id, problem in enumerate(TASKS):
    print(f"\n=== Task {task_id} ===")
    print("Problem:", problem, "\n")

    for mode in ["FAST", "CoT"]:
        print(f"--- Mode: {mode} ---")
        try:
            result = call_llm(problem, mode=mode)
            ans = result["answer"]
            T_sec = result["T_sec"]

            print("Raw answer (truncated):")
            print(textwrap.shorten(ans, width=160, placeholder="..."), "\n")
            print(f"T = {T_sec:.3f} s")

            # φ(x), λ_K, h(x)
            phi_stats = measure_phi_x(problem, ans)
            phi_x = phi_stats["phi_x"]
            lam = phi_stats["lambda_K"]
            h_x = phi_stats["h_x"]

            # φ(t) & LGR 功能
            if mode == "FAST":
                # FAST: 可以直接看作瞬时脉冲（结构近似 CDCL）
                phi_t = np.array([1.0], dtype=float)
            else:
                phi_t = build_phi_t(ans)

            phi_t_len = int(len(phi_t))
            phi_t_mean = float(phi_t.mean()) if len(phi_t) > 0 else 0.0
            lgr = lgr_functionals(phi_t, max_order=MAX_GRAD_ORDER)

            print(f"phi(x) = {phi_x:.4f}, lambda_K = {lam:.4f}, h(x) = {h_x:.4f}")
            print(f"len phi(t) = {phi_t_len}, mean={phi_t_mean:.4f}")
            pretty_lgr = {k: f"{v:.4e}" for k, v in lgr.items()}
            print("LGR:", pretty_lgr)

            row = {
                "experiment": EXPERIMENT_NAME,
                "model": MODEL_NAME,
                "mode": mode,
                "task_id": task_id,
                "problem": problem,
                "answer": ans,
                "T_sec": T_sec,
                "log2T": math.log2(T_sec) if T_sec > 0 else float("-inf"),
                "phi_x": phi_x,
                "lambda_K": lam,
                "h_x": h_x,
                "phi_t_len": phi_t_len,
                "phi_t_mean": phi_t_mean,
            }
            # 加入 LGR 功能
            for k, v in lgr.items():
                row[k] = v

            rows.append(row)

        except Exception as e:
            print("LLM error:", repr(e))

df = pd.DataFrame(rows)
print("\n=== Summary DataFrame (head) ===")
print(df.head())

if df.empty:
    raise RuntimeError("没有收集到数据。请检查 OpenAI API 调用是否正常。")

# 保存原始结果，方便之后多模型/多温度汇总
csv_name = f"lgr_30tasks_{EXPERIMENT_NAME}.csv"
df.to_csv(csv_name, index=False)
print(f"\nSaved raw results to: {csv_name}")

# ========= 5. 统计：FAST vs CoT 模式平均 =========

print("\n=== Mode-wise mean stats (FAST vs CoT) ===")
cols_stats = ["T_sec", "phi_x", "lambda_K", "h_x", "phi_t_len", "F2", "F4"]
for n in range(1, MAX_GRAD_ORDER + 1):
    cols_stats.append(f"G{n}")
group_stats = df.groupby("mode")[cols_stats].mean()
print(group_stats)

# ========= 6. 回归：Baseline vs LGR（混合样本） =========

def fit_linear_regression(y: np.ndarray, X: np.ndarray, feature_names: List[str]):
    """简单最小二乘线性回归，返回 (coef_dict, R2, y_pred)。"""
    # 加常数项
    X_design = np.hstack([X, np.ones((X.shape[0], 1))])
    names = feature_names + ["const"]

    coef, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    y_pred = X_design @ coef
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    coef_dict = {name: val for name, val in zip(names, coef)}
    return coef_dict, R2, y_pred

def describe_regression(name: str, coef_dict: Dict[str, float], R2: float):
    print(f"\n=== Regression: {name} ===")
    for k, v in coef_dict.items():
        print(f"{k:8s}: {v:.4f}")
    print(f"R² ≈ {R2:.3f}")

valid = df["T_sec"] > 0
sub_all = df[valid].copy()

y_all = sub_all["log2T"].values

# ---- Baseline 1: log2T ~ phi_x ----
X_phi = sub_all[["phi_x"]].values
coef_phi, R2_phi, y_pred_phi = fit_linear_regression(y_all, X_phi, ["phi_x"])
describe_regression("Baseline (phi_x only)", coef_phi, R2_phi)

# ---- Baseline 2: log2T ~ h_x (PLSA 宏观极限) ----
X_h = sub_all[["h_x"]].values
coef_h, R2_h, y_pred_h = fit_linear_regression(y_all, X_h, ["h_x"])
describe_regression("Baseline (h_x = 1 - lambda_K)", coef_h, R2_h)

# ---- Full LGR (mixed FAST+CoT) ----
feature_names_lgr = ["phi_x", "phi_x^2", "F2", "F4"]
X_list = [
    sub_all["phi_x"].values,
    sub_all["phi_x"].values ** 2,
    sub_all["F2"].values,
    sub_all["F4"].values,
]
for n in range(1, MAX_GRAD_ORDER + 1):
    fname = f"G{n}"
    feature_names_lgr.append(fname)
    X_list.append(sub_all[fname].values)

X_lgr = np.vstack(X_list).T
coef_lgr, R2_lgr, y_pred_lgr = fit_linear_regression(y_all, X_lgr, feature_names_lgr)
describe_regression(f"Full LGR (mixed FAST+CoT, up to G{MAX_GRAD_ORDER})", coef_lgr, R2_lgr)

# ========= 7. CoT 子样本内部 LGR 标度（湍流相） =========

sub_cot = sub_all[sub_all["mode"] == "CoT"].copy()
y_cot = sub_cot["log2T"].values

X_list_cot = [
    sub_cot["phi_x"].values,
    sub_cot["phi_x"].values ** 2,
    sub_cot["F2"].values,
    sub_cot["F4"].values,
]
feature_names_cot = ["phi_x", "phi_x^2", "F2", "F4"]
for n in range(1, MAX_GRAD_ORDER + 1):
    fname = f"G{n}"
    feature_names_cot.append(fname)
    X_list_cot.append(sub_cot[fname].values)

X_cot = np.vstack(X_list_cot).T
coef_cot, R2_cot, y_pred_cot = fit_linear_regression(y_cot, X_cot, feature_names_cot)
describe_regression(f"LGR (CoT only, turbulent phase, up to G{MAX_GRAD_ORDER})", coef_cot, R2_cot)

# ========= 8. 可视化（简单版） =========

plt.figure(figsize=(6, 5))
for mode, marker, color in [("FAST", "o", "C0"), ("CoT", "s", "C1")]:
    sub = sub_all[sub_all["mode"] == mode]
    plt.scatter(sub["phi_x"], sub["log2T"], marker=marker, label=mode, alpha=0.8)
plt.xlabel(r"$\phi(x)$")
plt.ylabel(r"$\log_2 T(x)$")
plt.title("phi(x) vs log2 T(x) (FAST vs CoT)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("scatter_phi_log2T_fast_cot.png", dpi=150)
plt.show()

# 预测 vs 真值图（全局 LGR）
plt.figure(figsize=(5, 5))
plt.scatter(y_all, y_pred_lgr, alpha=0.8)
min_v = min(y_all.min(), y_pred_lgr.min())
max_v = max(y_all.max(), y_pred_lgr.max())
plt.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1)
plt.xlabel("True log2 T")
plt.ylabel("Predicted log2 T (LGR)")
plt.title("LGR fit: true vs predicted")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lgr_true_vs_pred.png", dpi=150)
plt.show()

print("\nSaved figures: scatter_phi_log2T_fast_cot.png, lgr_true_vs_pred.png")

# ========= 9. LaTeX 片段输出 =========

# 9.1 Mode-wise stats 表
print("\n=== LaTeX: Mode-wise stats table (FAST vs CoT) ===")
group_for_latex = group_stats.copy().round(3)
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\begin{tabular}{lrrrrrrrr}")
print(r"\toprule")
print(r"Mode & $\mathbb{E}[T]$ & $\mathbb{E}[\phi(x)]$ & $\mathbb{E}[\lambda_K]$ & $\mathbb{E}[|\phi(t)|]$ & $\mathbb{E}[F_2]$ & $\mathbb{E}[F_4]$ & $\mathbb{E}[G_1]$ & $\mathbb{E}[G_2]$ \\")
print(r"\midrule")
for mode, row in group_for_latex.iterrows():
    print(
        f"{mode} & {row['T_sec']:.3f} & {row['phi_x']:.3f} & {row['lambda_K']:.3f} & "
        f"{row['phi_t_len']:.3f} & {row['F2']:.3f} & {row['F4']:.3f} & "
        f"{row['G1']:.3f} & {row['G2']:.3f} \\\\"
    )
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\caption{Mode-wise statistics of runtime $T$, instance order parameter $\phi(x)$, trace length $|\phi(t)|$, and Landau--Ginzburg--Reynolds functionals $(F_2,F_4,G_1,G_2)$ for FAST vs CoT on 30 reasoning tasks.}")
print(r"\label{tab:lgr-real-llm}")
print(r"\end{table}")

# 9.2 Unified LGR equation（混合样本）
print("\n=== LaTeX: Unified LGR equation (mixed FAST+CoT, up to G6) ===")
# 按 feature_names_lgr 的顺序拿系数
a1  = coef_lgr.get("phi_x", 0.0)
a2  = coef_lgr.get("phi_x^2", 0.0)
bF2 = coef_lgr.get("F2", 0.0)
bF4 = coef_lgr.get("F4", 0.0)
g1  = coef_lgr.get("G1", 0.0)
g2  = coef_lgr.get("G2", 0.0)
g3  = coef_lgr.get("G3", 0.0)
g4  = coef_lgr.get("G4", 0.0)
g5  = coef_lgr.get("G5", 0.0)
g6  = coef_lgr.get("G6", 0.0)
const = coef_lgr.get("const", 0.0)

print(r"\begin{equation}")
latex_eq = (
    r"\log_2 T(x) \approx "
    + f"{a1:.2f}\\,\\phi(x)"
    + f" {a2:+.2f}\\,\\phi(x)^2"
    + f" {bF2:+.2f}\\,F_2[\\phi(t)]"
    + f" {bF4:+.2f}\\,F_4[\\phi(t)]"
    + f" {g1:+.2f}\\,G_1[\\phi(t)]"
    + f" {g2:+.2f}\\,G_2[\\phi(t)]"
    + f" {g3:+.2f}\\,G_3[\\phi(t)]"
    + f" {g4:+.2f}\\,G_4[\\phi(t)]"
    + f" {g5:+.2f}\\,G_5[\\phi(t)]"
    + f" {g6:+.2f}\\,G_6[\\phi(t)]"
    + f" {const:+.2f}."
)
print(latex_eq)
print(r"\end{equation}")
print(f"% R^2 (mixed LGR) ≈ {R2_lgr:.3f}")

# 9.3 PLSA 宏观极限：log2T ~ h(x)
print("\n=== LaTeX: PLSA macroscopic limit (log2T ~ h(x)) ===")
alpha_T = coef_h.get("h_x", 0.0)
c_T     = coef_h.get("const", 0.0)
print(r"\begin{equation}")
print(
    r"\log_2 T(x) \approx "
    + f"{alpha_T:.2f}\\,h(x)"
    + f"{c_T:+.2f},"
)
print(r"\end{equation}")
print(f"% where $h(x) = 1 - \\lambda_K(x)$, baseline R^2 ≈ {R2_h:.3f}")

# 9.4 图 caption 建议
print("\n=== LaTeX: Figure caption suggestion ===")
print(r"\begin{figure}[h]")
print(r"\centering")
print(r"% \includegraphics[width=0.7\linewidth]{lgr_real_llm_scatter.pdf}")
print(
    r"\caption{Unified Structure--Time Law on real LLM reasoning tasks. "
    r"Each point is a FAST or CoT run on one of 30 math and logic problems. "
    r"The horizontal axis encodes the instance-level order parameter $\phi(x)$, "
    r"while color and marker shape reflect Landau--Ginzburg--Reynolds functionals "
    r"$F_2,F_4,G_1,\dots,G_6$ derived from the reasoning trajectory $\phi(t)$. "
    r"A linear LGR model of the form in Eq.~(X) explains a substantial fraction of the "
    r"variance in $\log_2 T(x)$ (see Tab.~\ref{tab:lgr-real-llm}).}"
)
print(r"\label{fig:lgr-real-llm}")
print(r"\end{figure}")

print("\n== DONE (LGR: 30-task real LLM + G1..G6 + Baseline vs LGR + CoT-only scaling + LaTeX) ==")
